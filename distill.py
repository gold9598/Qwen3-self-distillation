from modeling_qwen3_student import Qwen3Modified
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datasets import load_dataset
import torch

def load_open_orca(batch_size: int = 1, split: str = "train"):
    """Yield batches of text from the OpenOrca dataset.

    Parameters
    ----------
    batch_size: int, optional
        Number of samples to return per iteration.
    split: str, optional
        Dataset split to stream from.
    """
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

def main():
    data_loader = load_open_orca(batch_size=2, split="train")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "0GB", 1: "80GB", 2: "80GB"},
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)

    student = Qwen3Modified("Qwen/Qwen3-1.7B", device)

    for name, param in student.model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    trainable_params = [p for p in student.model.parameters() if p.requires_grad]
    
    if not trainable_params:
        if hasattr(student.model, "lm_head"):
            for p in student.model.lm_head.parameters():
                p.requires_grad = True
            trainable_params = list(student.model.lm_head.parameters())
        else:
            raise ValueError(
                "optimizer got an empty parameter list because no parameters were "
                "marked as requiring gradients"
            )
    
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

    teacher.eval()
    student.model.train()

    for batch in data_loader:
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)

        student_outputs = student.model(**inputs)
        
        loss = F.kl_div(student_outputs.logits, teacher_outputs.logits, reduction="batchmean")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__=="__main__":
    main()
