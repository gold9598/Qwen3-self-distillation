from datasets import load_dataset, concatenate_datasets
import torch.nn.functional as F

# -------- MMLU HELPER FUNCTIONS
CHOICES = ["A", "B", "C", "D"]
BATCH_SIZE       = 1          # adjust to GPU RAM
MAX_NEW_TOKENS   = 32768
SUBJECTS = [  # taken directly from the dataset script
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

def fmt_example(ex: dict, include_answer: bool) -> str:
    """
    Returns one question block in plain text.
    When include_answer=True (few-shot shots) we append 'Answer: X'.
    """
    stem    = ex["question"].strip()
    options = ex["choices"] if "choices" in ex else [ex[k] for k in CHOICES]
    block   = stem + "\n" + "\n".join(f"{l}. {opt.strip()}" for l, opt in zip(CHOICES, options))

    if include_answer:
        gold = ex["answer"]
        letter = CHOICES[gold] if isinstance(gold, int) else gold.strip().upper()
        return f"{block}\nAnswer: {letter}\n\n"
    return block


def build_prompt(fewshot: list[dict], test_ex: dict, tokenizer) -> str:
    """
    • Builds a messages list → chat template → final prompt string.
    • The last message includes the evaluation instruction recommended by Qwen-3.
    """
    # 1. Few-shot context (each ends with 'Answer: X')
    context = "".join(fmt_example(s, True) for s in fewshot)

    # 2. Test question without answer
    question = fmt_example(test_ex, False)

    # 3. Extra instruction for standardised output
    instr = (
        # "\n\nPlease think step by step. "
        '\n\nShow ONLY the letter you choose in a JSON object like {"answer": "C"}.'
    )

    user_msg = {"role": "user", "content": context + question + instr}

    # 4. Convert to the actual prompt string that Qwen-3 expects
    prompt = tokenizer.apply_chat_template(
        [user_msg],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking= False  # True by default
    )
    return prompt

def load_mmlu_full(split="test"):
    """Return one `datasets.Dataset` containing every MMLU subject."""
    parts = []
    for sub in SUBJECTS:
        ds_sub = load_dataset("lukaemon/mmlu", sub, split=split)   # or "cais/mmlu"
        ds_sub = ds_sub.rename_columns({"input": "question", "target": "answer"})  # harmonise names
        ds_sub = ds_sub.add_column("subject", [sub] * len(ds_sub))
        parts.append(ds_sub)
    return concatenate_datasets(parts)
