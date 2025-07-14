from typing import Callable, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
import torch
import json
import torch.nn.functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, repeat_kv

def bpmax(x, rd, p=5, c=5):
    base = (x + c)
    return (base / rd).pow(p)

def eager_attention_forward(
        module, query, key, value, attention_mask: Optional[torch.Tensor], scaling, dropout=0.0, p=8, c=7, **kwargs
    ):
    # Original PowerFormer code snippet follows:
    # batch_ret = torch.max(torch.sum(x,dim=-1,keepdims=True),dim=0,keepdims=True).values
    key_states   = repeat_kv(key,   module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # attn_output = torch.matmul(attn_weights, value_states)
    # attn_output = attn_output.transpose(1, 2).contiguous()

    # return attn_output, attn_weights

    if module.training:
        module.rd = (attn_weights+c) \
                    .sum(dim=-1) \
                    .max(dim=0, keepdim=True) \
                    .values.unsqueeze(-1) \

        # kl_loss = nn.KLDivLoss(reduction="batchmean")
        # print(torch.min(attn_weights), torch.max(attn_weights))

        attn_weights_approx = bpmax(attn_weights, rd=module.rd[:, :, :attn_weights.size(2), :], p=p, c=c)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        module.attn_softmax_loss = torch.sum((attn_weights_approx-attn_weights).view(-1))
        # print(torch.min(attn_weights_approx), torch.max(attn_weights_approx), torch.sum(attn_weights, dim=-1))
        #  module.attn_softmax_loss = kl_loss(torch.log(attn_weights_approx), attn_weights)

        # attn_diff = attn_weights_approx - attn_weights
        # attn_diff = attn_diff.view(-1)
        # module.attn_softmax_loss = torch.sum( \
        #                             ( \
        #                             torch.sum( \
        #                             attn_diff.view(attn_diff.size(0), attn_diff.size(1), -1), dim = -1)))

    else:
        attn_weights = bpmax(attn_weights, rd=module.rd[:, :, :attn_weights.size(2), :], p=p, c=c)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class Qwen3RMSNormModified(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)).to('cuda:1')
        self.variance_epsilon = eps
        self.denominator = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        if self.training:
            self.denominator = torch.rsqrt(variance + self.variance_epsilon).detach()

        hidden_states = hidden_states * self.denominator[:, :hidden_states.size(1), ...]

        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class Qwen3AttentionModified(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.rd = None
        self.training = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias, device='cuda:1'
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device='cuda:1'
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias, device='cuda:1'
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias, device='cuda:1'
        )
        self.q_norm = Qwen3RMSNormModified(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNormModified(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
