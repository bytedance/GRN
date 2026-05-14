import json
import math
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import tqdm
from timm.models import register_model
from torch.nn.attention.flex_attention import flex_attention

import grn.utils_t2iv.dist as dist
from grn.models.basic import FastRMSNorm, SelfAttnBlock
from grn.models.flex_attn_mask import build_flex_attn_func
from grn.models.rope import precompute_rope3d_freqs_grid
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils_t2iv.dist import for_visualize
from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import sp_gather_sequence_by_dim, sp_split_sequence_by_dim


class MultipleLayers(nn.Module):
    """A sequential container for a chunk of multiple transformer blocks."""

    def __init__(self, layers: List[nn.Module], num_blocks: int, start_index: int):
        super().__init__()
        self.module = nn.ModuleList([
            layers[i] for i in range(start_index, start_index + num_blocks)
        ])

    def forward(
        self, x: torch.Tensor, e0: Optional[torch.Tensor], 
        attn_bias_or_two_vector: Optional[Any], attn_fn: Optional[Any] = None, 
        checkpointing_full_block: bool = False, rope2d_freqs_grid: Optional[torch.Tensor] = None, 
        scale_ind: Optional[Any] = None, context_info: Optional[Any] = None, 
        last_diffusion_step: bool = True, ref_text_scale_inds: Optional[List[Any]] = None, 
        use_cfg: bool = False, split_cond_uncond: Optional[List[Any]] = None
    ) -> torch.Tensor:
        if ref_text_scale_inds is None:
            ref_text_scale_inds = []
        if split_cond_uncond is None:
            split_cond_uncond = []

        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(
                    m, h, e0, attn_bias_or_two_vector, attn_fn, 
                    rope2d_freqs_grid, scale_ind, context_info, 
                    last_diffusion_step, ref_text_scale_inds, 
                    use_cfg, split_cond_uncond, use_reentrant=False
                )
            else:
                h = m(
                    h, e0, attn_bias_or_two_vector, attn_fn, 
                    rope2d_freqs_grid, scale_ind, context_info, 
                    last_diffusion_step, ref_text_scale_inds, 
                    use_cfg, split_cond_uncond
                )
        return h


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal embeddings.
    
    Args:
        dim (int): Embedding dimension (must be even).
        position (torch.Tensor): Position tensor of shape [B, L].
        
    Returns:
        torch.Tensor: Embeddings of shape [B, L, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")
        
    half = dim // 2
    b, l = position.shape
    position = position.reshape(-1).type(torch.float64)

    sinusoid = torch.outer(
        position, 
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.reshape(b, l, dim)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        n, d = embedding.shape
        return embedding.reshape(n, 1, d)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def bld_to_bthwd(item: torch.Tensor, patch_time: int, patch_height: int, patch_width: int, apply_spatial_patchify: bool = False) -> torch.Tensor:
    """Reshape a sequence tensor to a spatial tensor."""
    batch_size = item.shape[0]
    return item.reshape(batch_size, patch_time, patch_height, patch_width, -1)


class FsqHead(nn.Module):
    """Classification head for Finite Scalar Quantization (FSQ)."""

    def __init__(self, hidden_dim: int, fsq_dim: int, fsq_lvl: int, use_ada_layer_norm: bool, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fsq_dim * fsq_lvl)
        self.norm = FastRMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.proj(self.norm(x))


class GRN(nn.Module):
    def __init__(
        self,
        vae_local: Any,
        arch: str = 'var',
        qwen_qkvo_bias: bool = False,
        text_channels: int = 0,
        text_maxlen: int = 0,
        embed_dim: int = 1024,
        depth: int = 16,
        num_key_value_heads: int = -1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        block_chunks: int = 1,
        checkpointing: Optional[str] = None,
        pad_to_multiplier: int = 0,
        use_flex_attn: bool = False,
        num_of_label_value: int = 2,
        rope2d_normalized_by_hw: int = 0,
        pn: Optional[str] = None,
        video_frames: int = 1,
        always_training_scales: int = 20,
        apply_spatial_patchify: int = 0,
        inference_mode: bool = False,
        other_args: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__()
        # 1. Model Configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.use_flex_attn = use_flex_attn
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        self.inference_mode = inference_mode
        self.other_args = other_args

        # 2. Embedding & Scale Configuration
        self.vae_embed_dim = vae_local.codebook_dim
        self.apply_spatial_patchify = apply_spatial_patchify
        self.text_channels = text_channels
        self.text_maxlen = text_maxlen
        self.is_text_to_image = text_channels != 0

        classifier_head_dim = other_args.detail_scale_dim
        classifier_head_lvl = other_args.detail_num_lvl
        hbq_round = other_args.hbq_round

        if other_args.refine_mode in ['ar_discrete_GRN_ind']:
            self.visual_embedding_in_dim = vae_local.codebook_dim * (2**hbq_round)
            classifier_head_dim = vae_local.codebook_dim
        elif other_args.refine_mode in ['ar_discrete_GRN_bit']:
            self.visual_embedding_in_dim = hbq_round * vae_local.codebook_dim * 2
            classifier_head_dim = hbq_round * vae_local.codebook_dim
        else:
            self.visual_embedding_in_dim = vae_local.codebook_dim

        if self.apply_spatial_patchify:
            self.visual_embedding_in_dim *= 4

        # 3. Dynamic Resolution & Video Specifics
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.pn = pn
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames
        )
        self.train_h_div_w_list = self.h_div_w_templates
        self.image_scale_repetition = json.loads(other_args.image_scale_repetition)
        self.video_scale_repetition = json.loads(other_args.video_scale_repetition)

        print(f"Arch: {arch}, pn: {self.pn}, num_of_label_value: {self.num_of_label_value}, "
              f"rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}")
        print(f"train_h_div_w_list: {self.train_h_div_w_list}, "
              f"image_scale_repetition: {self.image_scale_repetition}, "
              f"video_scale_repetition: {self.video_scale_repetition}")

        # 4. Utilities
        self.entrophy_statistics = []
        self.top_p, self.top_k = 1.0, 100
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.infer_ts = None

        # 5. Model Components (Projections, Embeddings)
        self.norm0_cond = nn.Identity()
        self.text_proj = nn.Linear(self.text_channels, self.embed_dim)

        if self.other_args.use_ada_layer_norm:
            self.scale_or_time_dim = 256
            self.scale_or_time_embedding = nn.Sequential(
                nn.Linear(self.scale_or_time_dim, self.embed_dim), nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.scale_or_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim * 6))

        tmp_h_div_w_template = self.train_h_div_w_list[0]
        self.scales_in_one_clip = self.dynamic_resolution_h_w[tmp_h_div_w_template][self.pn]['scales_in_one_clip']

        # RoPE grid initialization
        with torch.amp.autocast('cuda', dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                pad_to_multiplier=self.pad_to_multiplier, 
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010, # never used
                max_frames=int(self.video_frames / other_args.temporal_compress_rate + 1),
                max_height=1800 // 8, 
                max_width=1800 // 8,
                text_maxlen=self.text_maxlen,
                pn=self.pn,
                args=other_args,
            )

        self.word_embed = nn.Linear(self.visual_embedding_in_dim, self.embed_dim)
        self.head = FsqHead(
            hidden_dim=self.embed_dim, 
            fsq_dim=classifier_head_dim, 
            fsq_lvl=classifier_head_lvl, 
            use_ada_layer_norm=other_args.use_ada_layer_norm,
        )

        if other_args.add_scale_token > 0:
            self.pt_embedder = TimestepEmbedder(self.embed_dim)
        if other_args.add_class_token > 0:
            self.class_tokens = nn.Parameter(torch.randn(1001, 1, other_args.add_class_token, self.embed_dim))
            print(f"class_tokens shape: {self.class_tokens.shape}")

        # 6. Transformer Blocks
        self.attn_fn_compile_dict = {}
        if self.use_flex_attn:
            self.flex_attention = torch.compile(flex_attention)

        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.embed_dim, 
                num_heads=num_heads, 
                num_key_value_heads=num_key_value_heads, 
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                qwen_qkvo_bias=qwen_qkvo_bias,
                use_ada_layer_norm=other_args.use_ada_layer_norm,
            )
            self.unregistered_blocks.append(block)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // self.num_block_chunks
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth, "Depth must be divisible by block_chunks"
        
        self.block_chunks = nn.ModuleList([
            MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i * self.num_blocks_in_a_chunk)
            for i in range(self.num_block_chunks)
        ])

        print(f"    [Model Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, "
              f"mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}")
        print(f"    drop_path_rate={drop_path_rate:g}", end='\n\n', flush=True)
        
    def get_loss_acc(
        self,
        hidden_states: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
        sequence_packing_scales: List[List[Tuple[int, int, int]]],
        gt: List[torch.Tensor],
        other_info_by_scale: List[Dict[str, Any]],
        return_last_hidden_states: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate loss and accuracy for the predicted logits.
        
        Args:
            hidden_states: shaped (B, L, C)
            hidden_states_mask: Optional mask for hidden states
            e: scale or time embeddings
            sequence_packing_scales: List of scales for sequence packing
            gt: Ground truth labels
            other_info_by_scale: Meta information for each scale
            return_last_hidden_states: Whether to return the last hidden states
            
        Returns:
            Tuple of (logits_norm, loss_list, acc_list)
        """
        logits_norm = []
        logits_full = self.head(hidden_states, e)
        global_token_ptr, global_scale_ptr = 0, 0
        loss_list, acc_list = [], []

        for pack_scales in sequence_packing_scales:
            for pt, ph, pw in pack_scales:
                mul_pt_ph_pw = pt * ph * pw
                cur_bits = other_info_by_scale[global_scale_ptr]['cur_bits']
                cur_lvl = other_info_by_scale[global_scale_ptr]['cur_lvl']
                
                logits = logits_full[:, global_token_ptr:global_token_ptr + mul_pt_ph_pw]
                logits = logits.reshape(hidden_states.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl)
                logits = logits.permute(0, 3, 1, 2) # [1, num_of_label_value, mul_pt_ph_pw, d]
                
                logits_norm.append(logits.abs().mean())
                
                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none')[0] # [mul_pt_ph_pw, d]
                acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[0] # [mul_pt_ph_pw, d]
                
                loss_list.append(loss_this_scale.mean(-1))
                acc_list.append(acc_this_scale.mean(-1))
                
                global_scale_ptr += 1
                global_token_ptr += mul_pt_ph_pw + self.other_args.add_scale_token + self.other_args.add_class_token
                
        loss_tensor = torch.cat(loss_list) if loss_list else torch.tensor([], device=hidden_states.device)
        acc_tensor = torch.cat(acc_list) if acc_list else torch.tensor([], device=hidden_states.device)
        logits_norm_tensor = torch.stack(logits_norm).mean() if logits_norm else torch.tensor(0.0, device=hidden_states.device)
        
        return logits_norm_tensor, loss_tensor, acc_tensor
    
    def get_logits_during_infer(self, hidden_states: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits during inference."""
        return self.head(hidden_states.float(), e)
    
    def forward(
        self,
        label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]],
        x_BLC: torch.Tensor,
        visual_rope_cache: Optional[List[torch.Tensor]] = None,
        sequece_packing_scales: Optional[List[List[Tuple[int, int, int]]]] = None,
        super_scale_lengths: Optional[List[int]] = None,
        super_querysid_super_refsid: Optional[Any] = None,
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        gt_BL: Optional[List[torch.Tensor]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
        pad_seq_len: int = 0,
        scale_or_time_ids: Optional[torch.Tensor] = None,
        return_last_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Forward pass for the GRN model.
        
        Args:
            label_B_or_BLT: Text conditions or labels
            x_BLC: Input sequence hidden states
            visual_rope_cache: Cache for visual RoPE embeddings
            sequece_packing_scales: Scales for sequence packing
            super_scale_lengths: Lengths of super scales
            super_querysid_super_refsid: Query and reference scale IDs
            other_info_by_scale: Meta info for scales
            gt_BL: Ground truth
            x_BLC_mask: Mask for input sequence
            pad_seq_len: Padding length
            scale_or_time_ids: IDs for scale or time embeddings
            return_last_hidden_states: Whether to return last hidden states
            
        Returns:
            Tuple of (logits_norm, loss_list, acc_list, valid_sequence_ratio)
        """
        batch_size = 1 # sequence packing
        device = x_BLC[0].device

        # [1. get input sequence x_BLC]
        # word embedding
        sub_L_list = [item.shape[1] for item in x_BLC]
        cat_x_BLC = torch.cat(x_BLC, dim=1)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))
        
        # add scale tokens
        if self.other_args.add_scale_token > 0:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                pt_tokens = self.pt_embedder(torch.tensor([info['scale_token_id'] for info in other_info_by_scale], device=device))
            x_BLC = [torch.cat((x_BLC[ind], pt_tokens[ind].unsqueeze(0)), dim=1) for ind in range(len(x_BLC))]

        # add class tokens
        if self.other_args.add_class_token > 0:
            class_tokens = [self.class_tokens[info['class_token_id']] for info in other_info_by_scale]
            x_BLC = [torch.cat((x_BLC[ind], class_tokens[ind]), dim=1) for ind in range(len(x_BLC))]
        
        if self.other_args.add_class_token > 0: # c2i
            x_BLC = torch.cat(x_BLC, dim=1)
        else:
            # add text tokens
            kv_compact, lens, cu_seqlens_k, max_seqlen_k, _ = label_B_or_BLT
            with torch.amp.autocast('cuda', dtype=torch.float32):
                kv_compact = self.text_proj(kv_compact).contiguous()
            x_BLC = torch.cat(x_BLC+[kv_compact.unsqueeze(0)], dim=1)
        
        if pad_seq_len > 0:
            assert super_scale_lengths[-1] == pad_seq_len, f'{super_scale_lengths[-1]}!= {pad_seq_len}, attention will be wrong, this error is fatal!!!'
            x_BLC = F.pad(x_BLC, (0, 0, 0, pad_seq_len), value=0.0)
        valid_sequence_ratio = 1 - pad_seq_len / x_BLC.shape[1]
        assert self.use_flex_attn
        attn_bias_or_two_vector = None
        
        attn_fn = build_flex_attn_func(
            flex_attention=self.flex_attention,
            seq_l=x_BLC.shape[1],
            prefix_lens=lens,
            args=self.other_args,
            device=x_BLC.device,
            batch_size=B,
            heads=None,
            pad_seq_len=pad_seq_len,
            sequece_packing_scales=sequece_packing_scales,
            super_scale_lengths=super_scale_lengths,
            super_querysid_super_refsid=super_querysid_super_refsid,
        )

        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = []
        for i in range(len(visual_rope_cache)):
            rope_cache_list.append(visual_rope_cache[i])
            if self.other_args.add_scale_token > 0: # rope for pt tokens
                rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+self.other_args.add_scale_token])
            if self.other_args.add_class_token > 0: # rope for class tokens
                rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,552:552+self.other_args.add_class_token])
        for i in range(len(lens)):
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
        rope_cache = torch.cat(rope_cache_list, dim=4)
        if pad_seq_len > 0:
            rope_cache = F.pad(rope_cache, (0,0,0,pad_seq_len), 'constant', 0.)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'

        # calculate time or scale embeddings
        if self.other_args.use_ada_layer_norm:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.scale_or_time_embedding(sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()) # [1, visual_seq_len,] -> [1, visual_seq_len, 256] -> [1, visual_seq_len, C]
                if e.shape[1] < x_BLC.shape[1]:
                    e = F.pad(e, (0,0,0,x_BLC.shape[1]-e.shape[1]), 'constant', 0.) # [1, visual_seq_len, C] -> [1, L, C]
                e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.embed_dim)) # [1, L, C] -> [1, L, 6C] -> [1, L, 6, C]
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e, e0 = None, None
        
        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        for i, chunk in enumerate(self.block_chunks): # this path
            x_BLC = chunk(x=x_BLC, e0=e0, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        logits_norm, loss_list, acc_list = self.get_loss_acc(x_BLC, x_BLC_mask, e, sequece_packing_scales, gt_BL, other_info_by_scale, return_last_hidden_states)
        return logits_norm, loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
        negative_label_B_or_BLT: Optional[Tuple[torch.Tensor, ...]],
        use_cfg: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prepare text conditions for inference."""
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if use_cfg and negative_label_B_or_BLT is not None:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        prefix_tokens = kv_compact.unsqueeze(0)
        return prefix_tokens, lens
    
    def embeds_codes2input(self, last_stage: torch.Tensor) -> torch.Tensor:
        """Embed discrete codes into continuous input representations."""
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0, 2, 1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(last_stage) # norm0_ve is Identity
        return last_stage
    
    @torch.no_grad()
    def autoregressive_infer(
        self,
        vae: Optional[Any] = None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        negative_label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        g_seed: Optional[int] = None,
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
        gt_leak: int = 0,
        args: Optional[Any] = None,
        get_visual_rope_embeds: Optional[Any] = None,
        context_info: Optional[Any] = None,
        noise_list: Optional[List[torch.Tensor]] = None,
        class_token_id: int = 0,
        uncond_class_token_id: int = 1000,
        **kwargs: Any,
    ):
        """Autoregressive inference loop for the GRN model."""
        if cfg_list is None: cfg_list = []
        if tau_list is None: tau_list = []
        
        from grn.schedules.global_refine import shift_pt
        
        rng = None
        assert len(cfg_list) >= len(scale_schedule), "Not enough CFG values for scales"
        assert len(tau_list) >= len(scale_schedule), "Not enough tau values for scales"
        attn_mask = None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        text_scales = len(label_B_or_BLT) 
        total_steps = args.max_infer_steps + text_scales
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks
        use_cfg = any(np.array(cfg_list) != 1)
        cfg_interval = float(args.cfg_type.split('_')[-1])

        # text tokens forward
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to('cuda')
        for si, text_cond_tuple in enumerate(label_B_or_BLT):
            prefix_tokens, lens = self.prepare_text_conditions(text_cond_tuple, negative_label_B_or_BLT, use_cfg)
            device = prefix_tokens.device
            last_stage = prefix_tokens
            if use_cfg:
                rope_cache = torch.cat([self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[0]], self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[1]]], dim=4)
            else:
                rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[0]]
            
            e, e0 = None, None
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, e0=e0, attn_bias_or_two_vector=attn_mask, attn_fn=None, rope2d_freqs_grid=rope_cache, scale_ind=f't{si}', context_info=context_info, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=use_cfg, split_cond_uncond=lens)
            pbar.update(1)

        if args.refine_mode in ['ar_discrete_GRN_bit']:
            classes = 2
            this_scale_var_input = torch.zeros((1,args.detail_scale_dim*args.hbq_round,*scale_schedule[-1]), device=prefix_tokens.device, dtype=prefix_tokens.dtype)
        elif args.refine_mode in ['ar_discrete_GRN_ind']:
            classes = 2**args.hbq_round
            this_scale_var_input = torch.zeros((1,args.detail_scale_dim,*scale_schedule[-1]), device=prefix_tokens.device, dtype=prefix_tokens.dtype)
        
        scale_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+args.add_scale_token]
        class_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,552:552+args.add_class_token]
        if noise_list is not None:
            absolute_gt_labels = noise_list[0].to('cuda').permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
        assert len(scale_schedule) == 1
        si = 0
        pn = scale_schedule[0]
        pt, ph, pw = pn
        mul_pt_ph_pw = pt * ph * pw
        if args.add_class_token > 0:
            ref_text_scale_inds = []
        else:
            ref_text_scale_inds = [f't0']
        repeat_idx = -1
        cur_round_scales = args.max_infer_steps
        pure_rand_labels = torch.randint(low=0, high=classes, size=this_scale_var_input.shape, device=this_scale_var_input.device, dtype=this_scale_var_input.dtype)
        mixed_xt = pure_rand_labels
        this_scale_var_input = multiclass_labels2onehot_input(mixed_xt, classes) # [B,d*num_classes,t,h,w]
        if args.min_infer_steps > 0:
            min_infer_steps, max_infer_steps = args.min_infer_steps, cur_round_scales
        else:
            min_infer_steps, max_infer_steps = cur_round_scales, cur_round_scales
        next_pt = 0.
        decision_entrophy = None
        for cur_inner_round_si in range(cur_round_scales):
            cur_pt = next_pt
            if cur_inner_round_si == 0:
                self.entrophy_statistics.append([])
            repeat_idx += 1 # index scale tokens, very important
            scale_token_id = cur_pt # 0~1, float
            cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0
            rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, scale_schedule, si, 0, device, args, context_info, args.mapped_h_div_w_template)
            last_stage = self.embeds_codes2input(this_scale_var_input)
            if args.add_scale_token > 0:
                pt_tokens = self.pt_embedder(torch.tensor([scale_token_id], device=device))
                last_stage = torch.cat((last_stage, pt_tokens), dim=1)
                rope_cache = torch.cat((rope_cache, scale_token_rope_cache), dim=4)
            if args.add_class_token > 0:
                last_stage_cond = torch.cat((last_stage, self.class_tokens[class_token_id]), dim=1)
                last_stage_uncond = torch.cat((last_stage, self.class_tokens[uncond_class_token_id]), dim=1)
                rope_cache = torch.cat((rope_cache, class_token_rope_cache), dim=4)
            else:
                last_stage_cond = last_stage
                last_stage_uncond = last_stage
            if use_cfg:
                last_stage = torch.cat([last_stage_cond, last_stage_uncond], dim=1)
                rope_cache = torch.cat([rope_cache, rope_cache], dim=4)
                split_cond_uncond = [mul_pt_ph_pw+args.add_scale_token+args.add_class_token] * 2
            else:
                last_stage = last_stage_cond
                split_cond_uncond = [mul_pt_ph_pw+args.add_scale_token+args.add_class_token]
            e, e0 = None, None
            last_diffusion_step = False
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, e0=e0, attn_bias_or_two_vector=attn_mask, attn_fn=None, rope2d_freqs_grid=rope_cache, scale_ind=si, context_info=context_info, last_diffusion_step=last_diffusion_step, ref_text_scale_inds=ref_text_scale_inds, use_cfg=use_cfg, split_cond_uncond=split_cond_uncond)
            logits = self.get_logits_during_infer(last_stage, e=e)
            tmp_bs, tmp_seq_len = logits.shape[:2]
            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl) # [B,thw+...,d,2]
            pred_cond_logits = logits[:,:mul_pt_ph_pw] # [B,thw,d,2]
            pred_cond_probs = pred_cond_logits.softmax(-1) # [B,thw,d,2]
            categories = pred_cond_logits.shape[-1]
            entrophy = (-pred_cond_probs * torch.log2(pred_cond_probs)).sum(-1).mean().item() / np.log2(categories)

            pt_unshift = (cur_inner_round_si + 1) / (args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1., pt_unshift), args.snr_shift)
            next_pt = 1 - np.cos(np.pi/2*pt_shift)
            next_pt = next_pt * 0.999

            pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1) # [B,thw,d]
            pred_cond_labels = bld_to_bthwd(pred_cond_labels, pt, ph, pw)
            pred_uncond_logits = logits[:,(mul_pt_ph_pw+args.add_scale_token+args.add_class_token):(2*mul_pt_ph_pw+args.add_scale_token+args.add_class_token)] # [B,thw,d,2]
            if cfg != 1:
                pred_cfg_logits = pred_uncond_logits + cfg * (pred_cond_logits - pred_uncond_logits)
            else:
                pred_cfg_logits = pred_cond_logits
            pred_cfg_logits = pred_cfg_logits.mul(1/tau_list[si]) # [B,thw,d,2]
            pred_cfg_probs = pred_cfg_logits.softmax(dim=-1) # [B,thw,d,2]
            pred_cfg_labels = torch.argmax(pred_cfg_probs, dim=-1) # [B,thw,d]
            pred_cfg_labels = bld_to_bthwd(pred_cfg_labels, pt, ph, pw) # [B,t,h,w,d]
            pred_sample_labels = torch.multinomial(pred_cfg_probs.view(-1, args.detail_num_lvl), num_samples=1, replacement=True, generator=rng).view(tmp_bs, mul_pt_ph_pw, -1) # [B, thw,d]
            pred_sample_probs = torch.gather(pred_cfg_probs, dim=3, index=pred_sample_labels.unsqueeze(-1)).squeeze(-1) # [B,thw,d]
            pred_sample_probs = bld_to_bthwd(pred_sample_probs, pt, ph, pw) # [B,t,h,w,d]
            pred_sample_labels = bld_to_bthwd(pred_sample_labels, pt, ph, pw) # [B,t,h,w,d]

            assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100. # different ratio between prediciton and input
            pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.
            pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.
            mixed_xt_Bthwd_01 = mixed_xt.clone().permute(0,2,3,4,1)
            mixed_xt_Bthwd_01[mixed_xt_Bthwd_01<0] = 0
            pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.
            pred_cfg_flip_ratio = (pred_cfg_labels != mixed_xt_Bthwd_01).sum() / pred_cfg_labels.numel() * 100.
            pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.
            self.entrophy_statistics[-1].append({
                'cur_inner_round_si': cur_inner_round_si,
                'cur_pt': cur_pt,
                'entrophy': entrophy,
                'assume_flip_ratio': assume_flip_ratio,
                'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                'pred_cfg_flip_ratio': pred_cfg_flip_ratio.item(),
                'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
                'scale_token_id': scale_token_id,
                'class_token_id': class_token_id,
                'uncond_class_token_id': uncond_class_token_id,
                'pred_zero_ratio': pred_zero_ratio.item(),
                'pred_one_ratio': pred_one_ratio.item(),
                'meta': args.meta,
            })
            print(f'{repeat_idx=} {cur_inner_round_si=} {cur_pt=:.3f} {pred_sample_labels.shape=}')
            print(f'{assume_flip_ratio=:.2f}% {pred_cond_flip_ratio=:.2f}% {pred_cfg_flip_ratio=:.2f}% {pred_sample_flip_ratio=:.2f}%')
            if repeat_idx < gt_leak:
                gt_labels = absolute_gt_labels
                gt_flip_ratio = (gt_labels != mixed_xt_Bthwd_01).sum() / gt_labels.numel() * 100.
                gt_flip_ratio = gt_flip_ratio.item()
                pred_cond_acc = (gt_labels==pred_cond_labels).to(float).mean().item()
                pred_cfg_acc = (gt_labels==pred_cfg_labels).to(float).mean().item()
                pred_sample_acc = (gt_labels==pred_sample_labels).to(float).mean().item()
                print(f'{si=} {repeat_idx=} {entrophy=:.4f} {pred_cond_acc=:.4f} {pred_cfg_acc=:.4f} {pred_sample_acc=:.4f}')
                self.entrophy_statistics[-1][-1].update({
                    'gt_flip_ratio': gt_flip_ratio,
                    'pred_cond_acc': pred_cond_acc,
                    'pred_cfg_acc': pred_cfg_acc,
                    'pred_sample_acc': pred_sample_acc,
                })
                pred_sample_labels = gt_labels
            
            pred_sample_labels = pred_sample_labels.permute(0,4,1,2,3) # [B,t,h,w,d] -> [B,d,t,h,w]
            pred_sample_probs = pred_sample_probs.permute(0,4,1,2,3) # [B,t,h,w,d] -> [B,d,t,h,w]
            use_predict_mask = torch.rand(pred_sample_labels.shape, device=device) < next_pt
            mixed_xt = torch.where(use_predict_mask, pred_sample_labels, pure_rand_labels)
            this_scale_var_input = multiclass_labels2onehot_input(mixed_xt, classes) # [B,d*num_classes,t,h,w]
            pbar.update(1)
            if np.abs(cur_pt - 1) < 1e-6: break
            
        if args.refine_mode == 'ar_discrete_GRN_ind':
            from grn.utils_t2iv.hbq_util_t2iv import index_label2quant_features
            approx_signal = index_label2quant_features(pred_sample_labels, hbq_round=args.hbq_round)
        elif args.refine_mode == 'ar_discrete_GRN_bit':
            from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
            approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round) # [B, hbq_round_mul_d, t, h, w] -> [B,d,t,h,w]
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        img = self.summed_codes2images(vae, approx_signal)
        return ret, idx_Bl_list, img

    def summed_codes2images(self, vae: Any, summed_codes: torch.Tensor) -> torch.Tensor:
        """Decode summed codes into images using the VAE."""
        t1 = time.time()
        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 4, 1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))
        print(f"Decode takes {time.time() - t1:.1f}s")
        return img # bgr order

    @for_visualize
    def vis_key_params(self, ep: int) -> None:
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:       
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(self, **kwargs: Any) -> None:
        """Apply special initialization to specific layers."""
        std = 0.02
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def extra_repr(self) -> str:
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str) -> Any:
        raise NotImplementedError

TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def GRN0b(depth: int = 4, block_chunks: int = 2, embed_dim: int = 512, num_heads: int = 4, num_key_value_heads: int = 4, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def GRN2b(depth: int = 28, block_chunks: int = 7, embed_dim: int = 2304, num_heads: int = 18, num_key_value_heads: int = 18, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )