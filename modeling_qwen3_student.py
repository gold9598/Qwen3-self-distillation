import math
from typing import Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required to use the student model") from e


def bpmax(x: torch.Tensor, p: float = 5.0, c: float = 5.0, rd: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch Parameterized Max (BPMax) normalization.

    Args:
        x: Attention scores ``(batch, heads, seq_len, seq_len)``.
        p: Exponent used in the numerator.
        c: Additive constant.
        rd: Precomputed denominator. Required when ``training`` is ``False``.
        training: Switch between training and inference behavior.

    Returns:
        Tuple of ``(normalized_scores, denominator)``.
    """
    powered = (x + c).pow(p)
    if training:
        # Compute denominator per batch. Sum over the key dimension then take max over the batch.
        denom = powered.sum(-1).max()
    else:
        if rd is None:
            raise ValueError("rd must be provided during inference")
        denom = rd
    return powered / denom, denom


def eager_attention_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, p: float = 5.0, c: float = 5.0, rd: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simplified eager attention using BPMax instead of softmax."""
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_probs, rd_value = bpmax(scores, p=p, c=c, rd=rd, training=training)
    context = torch.matmul(attn_probs, value)
    return context, attn_probs, rd_value
