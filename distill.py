import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_open_orca(split: str = "train", text_field: str | None = None, batch_size: int = 1):
    """Yield batches from the Open-Orca dataset.

    Parameters
    ----------
    split:
        Dataset split to use.
    text_field:
        Column name containing the textual prompt. If ``None``, a suitable
        column is selected automatically.
    batch_size:
        Number of samples per batch.
    """

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required to load Open-Orca."
        ) from e

    ds = load_dataset("Open-Orca/OpenOrca", split=split)

    if text_field is None:
        for candidate in [
            "prompt",
            "question",
            "text",
        ]:
            if candidate in ds.column_names:
                text_field = candidate
                break
    if text_field is None or text_field not in ds.column_names:
        raise ValueError(
            f"Could not infer text field from columns: {ds.column_names}"
        )

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size][text_field]
        yield batch


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


def distill(teacher_name: str, student_name: str, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = AutoModelForCausalLM.from_pretrained(teacher_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    student = AutoModelForCausalLM.from_pretrained(student_name).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)

    teacher.eval()
    student.train()

    for batch in data_loader:
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            teacher_outputs = teacher(**inputs)
        student_outputs = student(**inputs)

        denominator = extract_denominator(teacher_outputs)

        loss = torch.nn.functional.kl_div(
            torch.log_softmax(student_outputs.logits / denominator, dim=-1),
            torch.softmax(teacher_outputs.logits / denominator, dim=-1),
            reduction="batchmean",
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return student


if __name__ == "__main__":
    loader = load_open_orca(batch_size=1)
    distill("Qwen/Qwen3-4B", "Qwen/Qwen3-4B", loader)
