# Qwen3-self-distillation

This repository demonstrates a minimal setup for distilling a Qwen3 model.
It includes an implementation of denominator extraction based on the Batch
Method and utilities for loading the [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
dataset.

The dataset itself is **not** bundled with this repository. It will be
downloaded on demand via the `datasets` library when running the script.
