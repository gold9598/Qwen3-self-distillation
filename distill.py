from datasets import load_dataset, concatenate_datasets
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import json
from lm_eval.utils import  setup_logging
from tqdm import tqdm
import torch.nn.functional as F
import re
import random

from qwen3_modified import *
from load_mmlu import *

JSON_RE = re.compile(r'["\']?answer["\']?\s*:\s*["\']?\s*([ABCD])', re.I)

# -- Distillation dataset : OpenOrca

def load_open_orca(batch_size: int = 1, split: str = "train"):
    dataset = load_dataset("Open-Orca/OpenOrca", split=split, streaming=True)

    batch = []
    for sample in dataset:
        prompt = " ".join(
            str(sample.get(key, ""))
            for key in ["system_prompt", "question"]
            if sample.get(key)
        ).strip()
        batch.append(prompt)
        if len(batch) == batch_size:
            yield list(batch)
            batch.clear()
    if batch:
        yield list(batch)

# -- main functionality start
# ds = load_dataset("lukaemon/mmlu", "all", split="test", trust_remote_code=True)    # ~5 700 Qs
ds = load_mmlu_full()
subjects = ds.unique("subject")

num_modified_layers = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        trust_remote_code=True,
        device_map="cuda:0",
)

student = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",
        trust_remote_code=True,
        device_map="cuda:1",
)

data_loader = load_open_orca(batch_size=1, split="train")

for decoder_index in range(len(student.model.layers)-num_modified_layers):
    for name, param in student.model.layers[decoder_index].named_parameters():
        param.requires_grad = False

for inverse_index in range(1,num_modified_layers+1):
    old_attn = student.model.layers[-inverse_index].self_attn
    new_attn = Qwen3AttentionModified(old_attn.config, old_attn.layer_idx)

    with torch.no_grad():
        new_attn.q_proj.weight.copy_(old_attn.q_proj.weight)
        new_attn.k_proj.weight.copy_(old_attn.k_proj.weight)
        new_attn.v_proj.weight.copy_(old_attn.v_proj.weight)
        new_attn.o_proj.weight.copy_(old_attn.o_proj.weight)

    del student.model.layers[-inverse_index].self_attn, old_attn
    student.model.layers[-inverse_index].self_attn = new_attn

    student.model.layers[-inverse_index].self_attn.training = True
    student.model.layers[-inverse_index].self_attn.q_norm_training = True
    student.model.layers[-inverse_index].self_attn.k_norm_training = True

teacher.eval()
student.train()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

need_optim = any(p.requires_grad for p in student.parameters())
optim      = torch.optim.AdamW(
                 (p for p in student.parameters() if p.requires_grad),
                 lr=5e-5
             ) if need_optim else None

cnt = 0

fewshot_dict = {}
for subj in ds.unique("subject"):
    # slice out all rows for this subject
    subj_rows = ds.filter(lambda e, s=subj: e["subject"] == s)
    # keep the first 1 as “shots” (MMLU paper setting)
    fewshot_dict[subj] = subj_rows.select(range(1))
full_list = [i for i in range(0, len(ds), BATCH_SIZE)]
sampled_list = full_list[::14]

for batch in data_loader:                # your Open-Orca loader
    attn_student_results = []
    attn_teacher_results = []
    enc_teacher = {k: v.to('cuda:0')
           for k, v in tokenizer(batch, return_tensors='pt',
                                 padding='max_length', truncation=True,
                                 max_length=512, padding_side='left').items()}

    with torch.no_grad():
        teacher_logits = teacher(**enc_teacher, use_cache=True, output_attentions=False).logits.detach().to('cuda:0')

    enc_student = {k: v.to('cuda:1')
           for k, v in tokenizer(batch, return_tensors='pt',
                                 padding='max_length', truncation=True,
                                 max_length=512, padding_side='left').items()}

    # forward through the student
    student_logits = student(**enc_student, use_cache=True).logits.to('cuda:0')
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    loss = kl_loss(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1))

    '''
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_logprob = F.log_softmax(student_logits, dim=-1)
    prod_probs = teacher_prob * student_logprob
    x = torch.sum(prod_probs, dim=-1).view(-1)
    loss = -torch.sum(x, dim=0)
    '''

    # for decoder in student.model.layers[-num_modified_layers:]:
    #     loss += decoder.self_attn.attn_softmax_loss.to('cuda:0')

    loss.backward()
    optim.step()
    optim.zero_grad()

    cnt+=1;

    # MMLU evaluation
    if cnt % 64 == 0:
        student.eval()

        for inverse_index in range(1,num_modified_layers):
            student.model.layers[-inverse_index].self_attn.training = False
            student.model.layers[-inverse_index].self_attn.q_norm_training = False
            student.model.layers[-inverse_index].self_attn.k_norm_training = False

        fewshot = {s: ds.filter(lambda e: e["subject"] == s).select(range(5))
                    for s in subjects}

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",
                                            trust_remote_code=True,
                                            local_files_only=True)
        tok.padding_side = "left"
        tok.pad_token    = tok.eos_token
        total=correct=0

        for start in sampled_list:
            batch_select  = ds.select(range(start, min(start + BATCH_SIZE, len(ds))))
            prompts = [
                build_prompt(fewshot_dict[ex["subject"]], ex, tok)
                for ex in batch_select                   # batch is a Dataset slice (dicts)
            ]

            enc = tok(prompts, return_tensors="pt", padding='max_length', truncation=True, max_length=500, padding_side='left').to('cuda:1')

            generated_ids = student.generate(**enc, max_new_tokens=512-500, do_sample=True,
                         temperature=0.7, top_p=0.8, top_k=20, min_p=0)

            out = [generated_ids[i][len(enc.input_ids[i]):].tolist() for i in range(len(generated_ids))]

            indices = []

            # parsing thinking content
            for o in out:
                try:
                    # rindex finding 151668 (</think>)
                    index = len(o) - o[::-1].index(151668)
                except ValueError:
                    index = 0
                indices.append(index)

            # MMLU does not require thinking content, just answer.
            out = [o[i:] for o, i in zip(out,indices)]

            completions = tokenizer.batch_decode(
                out,
                skip_special_tokens=True,
            )

            for ex, txt in zip(batch_select, completions):
                pred = (JSON_RE.search(txt).group(1).upper()) if JSON_RE.search(txt) else "?"
                gold_raw   = ex["answer"]
                gold = CHOICES[gold_raw] if isinstance(gold_raw, int) else gold_raw.strip().upper()
                correct += int(pred == gold)
                total   += 1
        print(f"sampled MMLU accuracy: {correct/total:.3%}  ({correct}/{total})\n")

        torch.cuda.empty_cache()

        for inverse_index in range(1,num_modified_layers):
            student.model.layers[-inverse_index].self_attn.training = True
            student.model.layers[-inverse_index].self_attn.q_norm_training = True
            student.model.layers[-inverse_index].self_attn.k_norm_training = True

        student.train()

    del teacher_logits, student_logits, enc_teacher, enc_student, loss
    torch.cuda.empty_cache()
