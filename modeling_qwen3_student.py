import math
from typing import Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, Qwen3Model, Qwen3Attention
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    p: float = 5.0,
    c: float = 5.0,
    rd: Optional[torch.Tensor] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_weights, module.rd = bpmax(attn_weights, p=p, c=c, rd=module.rd, training=True)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class Qwen3Modified():
    def __init__(self, name = "Qwen/Qwen3-1.7B", device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "0GB", 1: "80GB", 2: "80GB"},
            use_cache=False
        )

        self.model.model.layers[-1].self_attn.attention_interface = eager_attention_forward
        self.rd = [0.0 for _ in range(len(self.model.model.layers))]
