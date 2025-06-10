import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


def extract_denominator(outputs, p: float = 2.0, c: float = 0.0) -> torch.Tensor:
    """Compute the batch-level denominator as described in the paper.

    The paper's Batch Method replaces softmax with a power-based form and
    normalizes using the maximum summed activation across the batch. Here we
    implement that denominator extraction for a batch of logits.

    Args:
        outputs: Model outputs containing ``logits``.
        p: Exponent applied to the shifted logits.
        c: Constant shift applied before exponentiation.

    Returns:
        A scalar tensor representing the denominator ``R_d`` for the batch.
    """

    logits = outputs.logits  # (batch, seq_len, vocab)
    powered = (logits + c) ** p
    batch_sums = powered.sum(dim=-1)  # sum over vocab dimension
    denominator = batch_sums.max()  # max over batch and sequence
    return denominator.detach()


def bpmax(
    logits: torch.Tensor, rd: torch.Tensor, p: float = 2.0, c: float = 0.0
) -> torch.Tensor:
    """Apply the Batch Power-Max transformation."""
    return ((logits + c) ** p) / rd


def distill(teacher_name: str, student_name: str, data_loader):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "0GB", 1: "0GB", 2: "40GB", 3: "40GB"},
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)

    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "0GB", 1: "0GB", 2: "40GB", 3: "40GB"},
        use_cache=False,
    )

    for name, param in student.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    trainable_params = [p for p in student.parameters() if p.requires_grad]

    if not trainable_params:
        if hasattr(student, "lm_head"):
            for p in student.lm_head.parameters():
                p.requires_grad = True
            trainable_params = list(student.lm_head.parameters())
        else:
            raise ValueError(
                "optimizer got an empty parameter list because no parameters were "
                "marked as requiring gradients"
            )

    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)

    teacher.eval()
    student.train()

    rd_values = []

    for batch in data_loader:
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)

        denominator = extract_denominator(teacher_outputs)
        rd_values.append(denominator.cpu())

        student_outputs = student(**inputs)

        teacher_probs = bpmax(teacher_outputs.logits, denominator)
        student_probs = bpmax(student_outputs.logits, denominator)

        loss = torch.nn.functional.kl_div(
            torch.log(student_probs + 1e-12),
            teacher_probs,
            reduction="batchmean",
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    rd_mean = torch.stack(rd_values).mean()
    torch.save(rd_mean, "rd_value.pt")

    return student


if __name__ == "__main__":
    # Distil the Qwen3-4B model with itself using the OpenOrca dataset.
    loader = load_open_orca(batch_size=2, split="train")
    distill("Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B", loader)
