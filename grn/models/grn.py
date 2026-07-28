import json
import math
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import tqdm
from timm.models import register_model

import grn.utils_t2iv.dist as dist
from grn.models.basic import FastRMSNorm, SelfAttnBlock
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
        self, x, cu_seqlens, max_seqlen, e0: Optional[torch.Tensor], 
        attn_bias_or_two_vector: Optional[Any], attn_fn: Optional[Any] = None, 
        checkpointing_full_block: bool = False, rope2d_freqs_grid: Optional[torch.Tensor] = None, 
        scale_ind: Optional[Any] = None, context_info: Optional[Any] = None, 
        last_diffusion_step: bool = True, ref_text_scale_inds: Optional[List[Any]] = None, 
        use_cfg: bool = False, split_cond_uncond: Optional[List[Any]] = None
    ) -> torch.Tensor:
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(
                    m, h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn, 
                    rope2d_freqs_grid, scale_ind, context_info, 
                    last_diffusion_step, ref_text_scale_inds, 
                    use_cfg, split_cond_uncond, use_reentrant=False
                )
            else:
                h = m(
                    h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn, 
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
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def bld_to_bthwd(item: torch.Tensor, patch_time: int, patch_height: int, patch_width: int, apply_spatial_patchify: bool = False) -> torch.Tensor:
    """Reshape a sequence tensor to a spatial tensor."""
    batch_size = item.shape[0]
    return item.reshape(batch_size, patch_time, patch_height, patch_width, -1)


def build_attn_mask(seqlens, device):
    attn_mask = torch.zeros((1, 1, sum(seqlens), sum(seqlens)), dtype=torch.bool, device=device)
    q_start = 0
    for i in range(len(seqlens)):
        q_len = seqlens[i]
        q_end = q_start + q_len
        attn_mask[:, :, q_start:q_end, q_start:q_end] = True
        q_start = q_end
    return attn_mask


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
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames
        )
        self.train_h_div_w_list = self.h_div_w_templates
        print(f"train_h_div_w_list: {self.train_h_div_w_list}")

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

        # RoPE grid initialization
        with torch.amp.autocast('cuda', dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010, # never used
                max_frames=int(self.video_frames / other_args.temporal_compress_rate + 1),
                max_height=1800 // 8, 
                max_width=1800 // 8,
                text_maxlen=self.text_maxlen,
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

        # 6. Transformer Blocks
        self.attn_fn_compile_dict = {}
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
                predict_tokens = other_info_by_scale[global_scale_ptr]['predict_tokens']
                all_tokens = other_info_by_scale[global_scale_ptr]['all_tokens']
                logits = logits_full[:, global_token_ptr:global_token_ptr + predict_tokens]
                logits = logits.reshape(hidden_states.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl)
                logits = logits.permute(0, 3, 1, 2) # [1, num_of_label_value, mul_pt_ph_pw, d]
                
                logits_norm.append(logits.abs().mean())
                
                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none')[0] # [mul_pt_ph_pw, d]
                acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[0] # [mul_pt_ph_pw, d]
                
                loss_list.append(loss_this_scale.mean(-1))
                acc_list.append(acc_this_scale.mean(-1))
                
                global_scale_ptr += 1
                global_token_ptr += all_tokens
                
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
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        gt_BL: Optional[List[torch.Tensor]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
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
            other_info_by_scale: Meta info for scales
            gt_BL: Ground truth
            x_BLC_mask: Mask for input sequence
            scale_or_time_ids: IDs for scale or time embeddings
            return_last_hidden_states: Whether to return last hidden states
            
        Returns:
            Tuple of (logits_norm, loss_list, acc_list, valid_sequence_ratio)
        """
        device = x_BLC[0].device

        # [1. get input sequence x_BLC]
        # word embedding
        sub_L_list = [item.shape[1] for item in x_BLC]
        cat_x_BLC = torch.cat(x_BLC, dim=1)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))

        # text tokens embedding
        kv_compact, lens, cu_seqlens_k, max_seqlen_k, _ = label_B_or_BLT
        with torch.amp.autocast('cuda', dtype=torch.float32):
            kv_compact = self.text_proj(kv_compact).contiguous() # [sum(lens), C]
        kv_compact_splits = torch.split(kv_compact, lens, dim=0)

        # scale tokens embedding
        scale_token_ids = torch.tensor([info["scale_token_id"] for info in other_info_by_scale], device=device)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            pt_tokens = self.pt_embedder((scale_token_ids)) # [num_scales, C]
        
        # construct final X_BLC input, [visual token, text token, scale token]
        x_BLC_lists = []
        for i in range(len(x_BLC)):
            x_BLC_lists.extend([x_BLC[i], kv_compact_splits[i].unsqueeze(0), pt_tokens[i][None, None]])
        x_BLC = torch.cat(x_BLC_lists, dim=1)

        valid_sequence_ratio = x_BLC.shape[1] / self.other_args.train_max_token_len
        attn_fn, attn_bias_or_two_vector = None, None

        # calculate finalrope cache, [visual token, text token, scale token]
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = []
        for i in range(len(visual_rope_cache)):
            rope_cache_list.append(visual_rope_cache[i])
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+self.other_args.add_scale_token])
        rope_cache = torch.cat(rope_cache_list, dim=4) # (2, 1, 1, 1, seq_len, head_dim / 2)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, head_dim / 2) -> (2, 1, 1, seq_len, head_dim / 2) -> (2, 1, seq_len, 1, head_dim / 2)

        # calculate time or scale embeddings
        if self.other_args.use_ada_layer_norm:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.scale_or_time_embedding(sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()) # [1, visual_seq_len,] -> [1, visual_seq_len, 256] -> [1, visual_seq_len, C]
                if e.shape[1] < x_BLC.shape[1]:
                    e = F.pad(e, (0,0,0,x_BLC.shape[1]-e.shape[1]), 'constant', 0.) # [1, visual_seq_len, C] -> [1, L, C]
                e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.C)) # [1, L, C] -> [1, L, 6C] -> [1, L, 6, C]
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e, e0 = None, None
        
        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        cu_seqlens = torch.tensor([0]+super_scale_lengths, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(super_scale_lengths)
        for i, chunk in enumerate(self.block_chunks): # this path
            x_BLC = chunk(x=x_BLC, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

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
        if use_cfg:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, lens
    
    def embeds_codes2input(self, last_stage: torch.Tensor) -> torch.Tensor:
        """Embed discrete codes into continuous input representations."""
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0, 2, 1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(last_stage) # norm0_ve is Identity
        return last_stage
    
    def encode_first_frame(self, vae, image_path, tgt_h, tgt_w):
        from grn.utils_t2iv.infer import transform
        from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
        raw_video = [cv2.imread(image_path)[:,:,::-1]] # bgr
        img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in raw_video] # bgr to rgb
        img_T3HW = torch.stack(img_T3HW, 0) # [t,3,h,w]
        img_bcthw = img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        raw_features, _, _ = vae.encode_for_raw_features(img_bcthw.to('cuda'), scale_schedule=None, slice=True)
        labels = raw_feature2bit_label(raw_features[0], hbq_round=self.other_args.hbq_round) # [B, hbq_round * d, t, h, w]
        return labels

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
        noise_list: Optional[List[torch.Tensor]] = None,
        first_frame_condition: bool = False,
        first_frame_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Autoregressive inference loop for the GRN model."""
        if cfg_list is None: cfg_list = []
        if tau_list is None: tau_list = []
        
        from grn.schedules.global_refine import shift_pt
        
        rng = None
        assert len(cfg_list) >= len(scale_schedule), "Not enough CFG values for scales"
        assert len(tau_list) >= len(scale_schedule), "Not enough tau values for scales"

        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        total_steps = args.max_infer_steps
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks
        use_cfg = True
        cfg_interval = float(args.cfg_type.split('_')[-1])
        full_pt, ph, pw = scale_schedule[0]
        if first_frame_condition:
            pt = full_pt - 1
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), 'cuda', args.mapped_h_div_w_template, t_offset=1)
        else:
            pt = full_pt
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), 'cuda', args.mapped_h_div_w_template, t_offset=0)

        # text tokens forward
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to('cuda')
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT[0], negative_label_B_or_BLT, use_cfg)
        device = prefix_tokens.device
        infer_device, infer_dtype = prefix_tokens.device, prefix_tokens.dtype
        prefix_tokens = torch.split(prefix_tokens, lens, dim=0)
        rope_cache_text_cond = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[0]]
        rope_cache_text_uncond = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[1]]

        if args.refine_mode in ['ar_discrete_GRN_bit']:
            classes = 2
            labels_shape = (1,args.detail_scale_dim*args.hbq_round,pt,ph,pw)
        elif args.refine_mode in ['ar_discrete_GRN_index']:
            classes = 2**args.hbq_round
            labels_shape = (1,args.detail_scale_dim,pt,ph,pw)

        mul_pt_ph_pw = pt * ph * pw
        repeat_idx = -1
        scale_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+args.add_scale_token]
        if noise_list is not None:
            absolute_gt_labels = noise_list[0].to('cuda').permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
        assert len(scale_schedule) == 1
        if first_frame_condition:
            first_frame_labels = self.encode_first_frame(vae, first_frame_path, ph*16, pw*16)[:,:,:1] # [B,d,1,h,w]
            first_frame_tokens_cond = self.embeds_codes2input(multiclass_labels2onehot_input(first_frame_labels, classes))
            fist_frame_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (1, ph, pw), device, args.mapped_h_div_w_template, t_offset=0)
            visual_rope_cache = torch.cat((visual_rope_cache, fist_frame_rope_cache), dim=4)
            tmp_seqlens = [mul_pt_ph_pw + ph * pw + lens[0] + args.add_scale_token, mul_pt_ph_pw + lens[1] + ph * pw + args.add_scale_token]
        else:
            tmp_seqlens = [mul_pt_ph_pw+lens[0]+args.add_scale_token, mul_pt_ph_pw+lens[1]+args.add_scale_token]

        # [visual tokens, text tokens, pt tokens]
        rope_cache = torch.cat([visual_rope_cache, rope_cache_text_cond, scale_token_rope_cache, visual_rope_cache, rope_cache_text_uncond, scale_token_rope_cache], dim=4) # (2, 1, 1, 1, seq_len, dim / 2)
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, dim / 2) -> (2, 1, 1, seq_len, dim / 2) -> (2, 1, seq_len, 1, dim / 2)

        cu_seqlens = torch.tensor([0]+tmp_seqlens, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(tmp_seqlens)

        pure_rand_labels = torch.randint(low=0, high=classes, size=labels_shape, device=infer_device, dtype=infer_dtype)
        mixed_xt = pure_rand_labels
        next_pt = 0.
        attn_mask = build_attn_mask(tmp_seqlens, device) if args.use_slow_attn else None
        for cur_inner_round_si in range(args.max_infer_steps):
            cur_pt = next_pt
            is_last_step = np.abs(cur_pt - 1) < 0.02
            if cur_inner_round_si == 0:
                self.entrophy_statistics.append([])
            repeat_idx += 1 # index scale tokens, very important
            cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0
            last_stage = self.embeds_codes2input(multiclass_labels2onehot_input(mixed_xt, classes))
            pt_tokens = self.pt_embedder(torch.tensor([cur_pt], device=device)).unsqueeze(0)
            # [visual tokens, text tokens, pt tokens]
            if first_frame_condition:
                last_stage_cond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                last_stage_uncond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
            else:
                last_stage_cond = torch.cat((last_stage, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                last_stage_uncond = torch.cat((last_stage, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
            last_stage = torch.cat([last_stage_cond, last_stage_uncond], dim=1)

            e, e0 = None, None
            last_diffusion_step = False
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, attn_bias_or_two_vector=attn_mask, attn_fn=None, rope2d_freqs_grid=rope_cache, last_diffusion_step=last_diffusion_step)
            logits = self.get_logits_during_infer(last_stage, e=e)
            tmp_bs, tmp_seq_len = logits.shape[:2]
            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl) # [B,thw+...,d,2]
            pred_cond_logits = logits[:,:mul_pt_ph_pw] # [B,thw,d,2]
            pred_uncond_logits = logits[:,tmp_seqlens[0]:tmp_seqlens[0]+mul_pt_ph_pw] # [B,thw,d,2]
            pred_cond_probs = pred_cond_logits.softmax(-1) # [B,thw,d,2]
            categories = pred_cond_logits.shape[-1]
            entrophy = (-pred_cond_probs * torch.log2(pred_cond_probs)).sum(-1).mean().item() / np.log2(categories)

            pt_unshift = (cur_inner_round_si + 1) / (args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1., pt_unshift), args.snr_shift)
            next_pt = 1 - np.cos(np.pi/2*pt_shift)
            next_pt = next_pt * 0.95

            pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1) # [B,thw,d]
            pred_cond_labels = bld_to_bthwd(pred_cond_labels, pt, ph, pw)
            if cfg != 1:
                pred_cfg_logits = pred_uncond_logits + cfg * (pred_cond_logits - pred_uncond_logits)
            else:
                pred_cfg_logits = pred_cond_logits
            pred_cfg_logits = pred_cfg_logits.mul(1/tau_list[0]) # [B,thw,d,2]
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
                # 'cur_tau': cur_tau,
                # 'cur_cfg': cur_cfg,
                'entrophy': entrophy,
                'assume_flip_ratio': assume_flip_ratio,
                'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                'pred_cfg_flip_ratio': pred_cfg_flip_ratio.item(),
                'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
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
                print(f'{repeat_idx=} {entrophy=:.4f} {pred_cond_acc=:.4f} {pred_cfg_acc=:.4f} {pred_sample_acc=:.4f}')
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
            next_pt = use_predict_mask.float().mean().item()
            pbar.update(1)
            if is_last_step: break

        if first_frame_condition:
            pred_sample_labels = torch.cat((first_frame_labels, pred_sample_labels), dim=2)
         
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
        std = math.sqrt(1 / self.C / 3)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        residual_scale = 1 / math.sqrt(2 * self.depth)
        for block in self.unregistered_blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            block.mlp.down_proj.weight.data.mul_(residual_scale)
    
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

@register_model
def GRN8b(depth: int = 36, block_chunks: int = 6, embed_dim: int = 4096, num_heads: int = 32, num_key_value_heads: int = 8, mlp_ratio: float = 4, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def GRN8b(depth: int = 36, block_chunks: int = 6, embed_dim: int = 4096, num_heads: int = 32, num_key_value_heads: int = 32, mlp_ratio: float = 3, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
