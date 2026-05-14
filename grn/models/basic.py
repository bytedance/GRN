import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc

from grn.models.rope import apply_rotary_emb
from grn.utils_t2iv.sequence_parallel import sp_all_to_all, SequenceParallelManager as sp_manager

try:
    from flash_attn.cute import flash_attn_varlen_func
except:
    from flash_attn import flash_attn_varlen_func
                
# Import flash_attn's fused ops
try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight

def merge_states(states, splits, cfg):
    """
    pick key and value states for flash_attn_varlen_func
    Args:
        states: list of states
        splits: list of split sizes
        cfg: bool, use cfg or not"""
    if cfg:
        cond_len, uncond_len = 0, 0
        cond_states, uncond_states = [], []
        for stat_, split_ in zip(states, splits):
            cond, uncond = torch.split(stat_, split_, dim=2)
            cond_states.append(cond)
            uncond_states.append(uncond)
            cond_len += cond.shape[2]
            uncond_len += uncond.shape[2]
        return cond_states + uncond_states, [cond_len, uncond_len]
    else:
        cond_len = 0
        for stat_ in states:
            cond_len += stat_.shape[2]
        return states, [cond_len]

class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12, num_key_value_heads=-1,
        use_flex_attn=False, qwen_qkvo_bias=False, **kwargs,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_key_value_heads == -1 or num_heads % num_key_value_heads == 0
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_heads
        self.q_proj = nn.Linear(embed_dim, self.num_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads*self.head_dim, bias=qwen_qkvo_bias)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, embed_dim, bias=qwen_qkvo_bias)
        self.q_norm = FastRMSNorm(self.head_dim)
        self.k_norm = FastRMSNorm(self.head_dim)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = {}    # kv caching: only used during inference
        self.cached_v = {}    # kv caching: only used during inference
        self.cached_split_cond_uncond = {} # only used during inference

        self.use_flex_attn = use_flex_attn
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = {}
        self.cached_v = {}
        self.cached_split_cond_uncond = {}

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, rope2d_freqs_grid=[], scale_ind=0, context_info=None, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=False, split_cond_uncond=[], **kwargs):
        # x: fp32
        B, L, C = x.shape
        hidden_states = x
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # batch, num_key_value_heads, slen, head_dim

        if sp_manager.sp_on():
            # Headnum need to be sharded and L needs to be gathered
            # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
            sdim = 1
            gdim = 2
            L = L * sp_manager.get_sp_size()
            C = C // sp_manager.get_sp_size()
            query_states = sp_all_to_all(query_states, sdim, gdim)
            key_states = sp_all_to_all(key_states, sdim, gdim)
            value_states = sp_all_to_all(value_states, sdim, gdim)

        query_states, key_states = apply_rotary_emb(query_states, key_states, rope2d_freqs_grid)
        if self.caching:    # kv caching: only used during inference
            if last_diffusion_step:
                self.cached_k[scale_ind] = key_states
                self.cached_v[scale_ind] = value_states
                self.cached_split_cond_uncond[scale_ind] = split_cond_uncond
            if isinstance(scale_ind, int):
                ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                key_states = [self.cached_k[ind] for ind in ref_scale_inds] + [key_states]
                value_states = [self.cached_v[ind] for ind in ref_scale_inds] + [value_states]
                split_cond_uncond_list = [self.cached_split_cond_uncond[ind] for ind in ref_scale_inds] + [split_cond_uncond]
            else:
                key_states = [key_states]
                value_states = [value_states]
                split_cond_uncond_list = [split_cond_uncond]
            
            key_states, cu_seqlens_k = merge_states(key_states, split_cond_uncond_list, use_cfg)
            value_states, _ = merge_states(value_states, split_cond_uncond_list, use_cfg)

            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            # delete deprecated cached kv to save gpu memory
            if isinstance(scale_ind, int):
                ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
                for si in range(len(context_info)):
                    for ref_si in context_info[si]['ref_sids']:
                        ref_scale_2_last_use_scale[ref_si] = si
                for ref_si in range(scale_ind):
                    if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si] is not None):
                        tmpk, tmpv = self.cached_k[ref_si], self.cached_v[ref_si]
                        self.cached_k[ref_si], self.cached_v[ref_si] = None, None
                        del tmpk, tmpv

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        scale = self.head_dim**-0.5
        if self.use_flex_attn and attn_fn is not None:
            attn_output = attn_fn(query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale).transpose(1, 2).reshape(B, L, C)
        else:
            if attn_bias_or_two_vector is None:

                # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                attn_output = flash_attn_varlen_func(
                    q = query_states.permute([0,2,1,3]).contiguous().to(torch.bfloat16).squeeze(0),
                    k = key_states.permute([0,2,1,3]).contiguous().to(torch.bfloat16).squeeze(0),
                    v = value_states.permute([0,2,1,3]).contiguous().to(torch.bfloat16).squeeze(0),
                    cu_seqlens_q = torch.tensor([0] + split_cond_uncond, device=query_states.device).cumsum(-1).to(torch.int32),
                    cu_seqlens_k = torch.tensor([0] + cu_seqlens_k, device=query_states.device).cumsum(-1).to(torch.int32),
                    max_seqlen_q = max(split_cond_uncond),
                    max_seqlen_k = max(cu_seqlens_k),
                    softmax_scale=scale,
                )
                attn_output = attn_output.reshape(B, L, C).contiguous()
                # attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
            else:
                # slow attn
                attn_output = slow_attn(query=query_states, key=key_states, value=value_states, scale=scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)

            # fa3, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
            # from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
            # attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
            # attn_output = attn_output[0].reshape(B, L, C)

        if sp_manager.sp_on():
            # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
            sdim = 1
            gdim = 2
            attn_output = sp_all_to_all(attn_output, sdim, gdim)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
class SelfAttnBlock(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_key_value_heads, mlp_ratio=4.,
        use_flex_attn=False,
        qwen_qkvo_bias=False, use_ada_layer_norm=False, **kwargs,
    ):
        super(SelfAttnBlock, self).__init__()
        self.C = embed_dim
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_key_value_heads=num_key_value_heads,
            use_flex_attn=use_flex_attn, qwen_qkvo_bias=qwen_qkvo_bias, **kwargs,
        )
        self.mlp = Qwen3MLP(hidden_size=embed_dim, intermediate_size=round(embed_dim * mlp_ratio / 256) * 256)
        self.use_ada_layer_norm = use_ada_layer_norm
        if self.use_ada_layer_norm:
            self.modulation = nn.Parameter(torch.randn(1, 6, embed_dim) / embed_dim**0.5)
            self.input_layernorm = WanLayerNorm(embed_dim)
            self.post_attention_layernorm = WanLayerNorm(embed_dim)
        else:
            self.input_layernorm = FastRMSNorm(embed_dim)
            self.post_attention_layernorm = FastRMSNorm(embed_dim)
        
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, e0, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_ind=0, context_info=None, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=False, split_cond_uncond=[], **kwargs):
        # x: [B,L,C]
        # e0: [B, L, 6, C]
        if self.use_ada_layer_norm:
            assert e0.dtype == torch.float32
            e = e0
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            residual = x
            hidden_states = x
            hidden_states = self.input_layernorm(hidden_states).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond, **kwargs)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = residual + hidden_states * e[2].squeeze(2)
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
            hidden_states = self.mlp(hidden_states)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = residual + hidden_states * e[5].squeeze(2)
        else:
            residual = x
            hidden_states = x
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond, **kwargs)
            hidden_states = residual + hidden_states
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states
