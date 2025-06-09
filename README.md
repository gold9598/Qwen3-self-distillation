# Qwen3-self-distillation

This repository provides a minimal example of self-distilling the
`Qwen/Qwen3-4B` model with the **Batch Method** described in the included
excerpt. The teacher and student share the same architecture. The script
`distill.py` loads the [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
dataset via the Hugging Face `datasets` library and performs one-step
distillation while extracting the batch-wise denominator from the teacher
outputs.

```
python distill.py
```

The script relies on `trust_remote_code=True` when loading the Qwen3 model
because the architecture is not yet included in the released version of
`transformers`. Ensure you have an up-to-date installation of
`transformers` or install it from source.

The example uses a streaming data loader and is meant as a starting point
for experimenting with Batch Power‑Max and Batch Layer Normalization.
