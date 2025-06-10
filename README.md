# Qwen3-self-distillation

This repository provides a minimal example of self‑distilling the
`Qwen/Qwen3-4B` model with the **Batch Method** described in the included
excerpt. The teacher and student share the same architecture. The script
`distill.py` loads the [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
dataset via the Hugging Face `datasets` library and performs one-step
distillation. Only the student's final ``lm_head`` is trained and the
Batch Power‑Max transformation is used in place of softmax. During
training the batch-wide denominator ``R_d`` is recorded and saved to
``rd_value.pt`` for use during inference.

```
python distill.py
```

The example uses a streaming data loader and is meant as a starting point
for experimenting with Batch Power‑Max and Batch Layer Normalization.

## Requirements

This example requires `transformers>=4.40.0` to load the Qwen3 model.
If you encounter errors about `ALL_PARALLEL_STYLES`, upgrade transformers:

```bash
pip install --upgrade transformers
```
