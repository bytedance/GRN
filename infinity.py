"""
Definition of Infinity transformer model.
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any
import json
import os
import os.path as osp
import math

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
import numpy as np

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_fused_op_installed, SelfAttnBlock, FastRMSNorm, WanLayerNorm
from infinity.models.rope import precompute_rope2d_freqs_grid, precompute_rope3d_freqs_grid, precompute_rope4d_freqs_grid
from infinity.utils import misc
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.models.apg import normalized_guidance
from infinity.utils.sequence_parallel import sp_split_sequence_by_dim, sp_gather_sequence_by_dim, SequenceParallelManager as sp_manager
from infinity.models.hbq import multiclass_labels2onehot_input


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C

class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cu_seqlens, max_seqlen, e0, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, scale_ind=None, context_info=None, last_diffusion_step=True, ref_text_scale_inds=[], use_cfg=False, split_cond_uncond=[]):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cu_seqlens, max_seqlen, e0, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond, use_reentrant=False)
            else:
                h = m(h, cu_seqlens, max_seqlen, e0, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule, scale_ind, context_info, last_diffusion_step, ref_text_scale_inds, use_cfg, split_cond_uncond)
        return h

def sinusoidal_embedding_1d(dim, position):
    # position: [B, L,]
    # return: [B, L, dim]

    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    B, L = position.shape
    position = position.reshape(-1)
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.reshape(B, L, dim)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional. range [0,1]
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # N, D = embedding.shape
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def Bld2Bthwd(item, pt, ph, pw, apply_spatial_patchify=False):
    tmp_bs, tmp_seq_len = item.shape[:2]
    item = item.reshape(tmp_bs, pt, ph, pw, -1) # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
    if apply_spatial_patchify: # unpatchify operation
        item = item.permute(0,1,4,2,3) # [B, t, 4d, h, w]
        item = torch.nn.functional.pixel_shuffle(item, 2) # [B, t, d, 2h, 2w]
        item = item.permute(0,1,3,4,2) # [B, t, 2h, 2w, d]
    return item

class FsqHead(nn.Module):
    def __init__(self, hidden_dim, fsq_dim, fsq_lvl, use_ada_layer_norm, eps=1e-6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fsq_dim*fsq_lvl)
        self.use_ada_layer_norm = use_ada_layer_norm
        if self.use_ada_layer_norm:
            self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)
            self.norm = WanLayerNorm(hidden_dim, eps)
        else:
            self.norm = FastRMSNorm(hidden_dim)

    def forward(self, x, e):
        # x shape: [B, L, C]
        # e shape: [B, L, C]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            if self.use_ada_layer_norm:
                assert e.dtype == torch.float32
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
                x = self.proj(x)
            else:
                x = self.proj(self.norm(x))
            return x # [B, L, fsq_dim*fsq_lvl]

class Infinity(nn.Module):
    def __init__(
        self, vae_local,
        arch='var',                         # var or qwen
        qwen_qkvo_bias=False,               # qwen qwen_qkvo_bias
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, 
        num_key_value_heads=-1,
        num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        add_lvl_embeding_on_first_block=1,
        num_of_label_value=2,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        dynamic_scale_schedule='13_hand_craft',
        scale_embeds_num=128,
        other_args=None,
        **kwargs,
    ):
        super().__init__()
        # set hyperparameters
        self.C = embed_dim
        self.vae_embed_dim = vae_local.codebook_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        self.task_type = other_args.task_type

        classifier_head_dim = other_args.detail_scale_dim
        classifier_head_lvl = other_args.detail_num_lvl
        each_round_scales = np.array(json.loads(other_args.each_round_scales))
        magic_df_round = len(each_round_scales)
        if other_args.refine_mode in ['ar_discrete_GRN_index']:
            self.visual_embedding_in_dim = vae_local.codebook_dim * (2**magic_df_round)
            classifier_head_dim = vae_local.codebook_dim
        elif other_args.refine_mode in ['ar_discrete_GRN_bit']:
            self.visual_embedding_in_dim = magic_df_round * vae_local.codebook_dim  * 2
            classifier_head_dim = magic_df_round * vae_local.codebook_dim
        else:
            self.visual_embedding_in_dim = vae_local.codebook_dim
        if self.apply_spatial_patchify:
            self.visual_embedding_in_dim *= 4
        self.other_args = other_args
        self.mask_type = other_args.mask_type
        self.context_frames = other_args.context_frames
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames)
        self.num_of_label_value = num_of_label_value
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.image_batch_size = other_args.image_batch_size
        self.video_batch_size = other_args.video_batch_size
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.train_h_div_w_list = self.h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.entrophy_statistics = []

        assert add_lvl_embeding_on_first_block in [0,1]
        self.add_lvl_embeding_on_first_block = add_lvl_embeding_on_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.image_scale_repetition = json.loads(other_args.image_scale_repetition)
        self.video_scale_repetition = json.loads(other_args.video_scale_repetition)
        print(f'arch: {arch}, self.add_lvl_embeding_on_first_block: {self.add_lvl_embeding_on_first_block}, \
            self.num_of_label_value: {self.num_of_label_value}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw} \
            self.train_h_div_w_list: {self.train_h_div_w_list}, self.image_scale_repetition: {self.image_scale_repetition}, self.video_scale_repetition: {self.video_scale_repetition}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = 1., 100
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        self.infer_ts = None
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        self.selecting_idx = None
        self.num_classes = 0
        self.D = self.C
        
        self.text_proj = nn.Linear(self.Ct5, self.D)
        
        if self.other_args.use_ada_layer_norm:
            self.scale_or_time_dim = 256
            self.scale_or_time_embedding = nn.Sequential(
                nn.Linear(self.scale_or_time_dim, self.D), nn.SiLU(), nn.Linear(self.D, self.D),
            )
            self.scale_or_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.D, self.D * 6))

        if self.rope2d_each_sa_layer:
            if other_args.rope_type == '3d':
                tmp_h_div_w_template = self.train_h_div_w_list[0]
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    rope2d_freqs_grid = precompute_rope3d_freqs_grid(dim=self.C//self.num_heads,
                                                                    pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                                                                    activated_h_div_w_templates=self.train_h_div_w_list,
                                                                    steps_per_frame=other_args.steps_per_frame,
                                                                    max_scales=1000+10, # never used
                                                                    max_frames=int(self.video_frames/other_args.temporal_compress_rate+1),
                                                                    max_height=1800 // 8, max_width=1800 // 8,
                                                                    text_maxlen=self.text_maxlen,
                                                                    args=other_args,)
            else:
                raise ValueError(f'self.task_type == {self.task_type} unsupported!')
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        
         # init [input embedding layer] and [cls head]
        self.word_embed = nn.Linear(self.visual_embedding_in_dim, self.C)
        
        self.head = FsqHead(
            hidden_dim=self.C, 
            fsq_dim=classifier_head_dim, 
            fsq_lvl=classifier_head_lvl, 
            use_ada_layer_norm=other_args.use_ada_layer_norm,
        )
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        if other_args.add_scale_token > 0:
            self.pt_embedder = TimestepEmbedder(self.C)
        # fused norm
        fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.drop_path_rate = drop_path_rate
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, num_key_value_heads=num_key_value_heads, mlp_ratio=mlp_ratio, drop=drop_rate, tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                mask_type=other_args.mask_type, context_frames=other_args.context_frames, steps_per_frame=other_args.steps_per_frame,
                arch=self.arch,
                qwen_qkvo_bias=qwen_qkvo_bias,
                inject_sync=other_args.inject_sync,
                use_ada_layer_norm=other_args.use_ada_layer_norm,
            )
            # block.bfloat16()
            self.unregistered_blocks.append(block)
                
        # Diff loss
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        self.block_chunks = nn.ModuleList()
        for i in range(self.num_block_chunks):
            self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
    def get_loss_acc(self, x_BLC, x_BLC_mask, e, sequece_packing_scales, gt, other_info_by_scale, return_last_hidden_states):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        logits_norm = []
        logits_full = self.head(x_BLC, e)
        global_token_ptr, global_scale_ptr = 0, 0
        loss_list, acc_list = [], []
        for i in range(len(sequece_packing_scales)):
            for j in range(len(sequece_packing_scales[i])):
                pt, ph, pw = sequece_packing_scales[i][j]
                mul_pt_ph_pw = pt * ph * pw
                cur_bits = other_info_by_scale[global_scale_ptr]['cur_bits']
                cur_lvl = other_info_by_scale[global_scale_ptr]['cur_lvl']
                predict_tokens = other_info_by_scale[global_scale_ptr]['predict_tokens']
                all_tokens = other_info_by_scale[global_scale_ptr]['all_tokens']
                logits = logits_full[:,global_token_ptr:global_token_ptr+predict_tokens]
                logits = logits.reshape(x_BLC.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl)
                logits = logits.permute(0,3,1,2) # [1, mul_pt_ph_pw, d, num_of_label_value] -> [1, num_of_label_value, mul_pt_ph_pw, d]
                logits_norm.append(logits.abs().mean())
                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                classes = logits.shape[1]
                if classes > 1:
                    loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none')[0] # [mul_pt_ph_pw, d]
                    acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[0] # [mul_pt_ph_pw, d]
                    loss_this_scale = loss_this_scale.mean(-1)
                    acc_this_scale = acc_this_scale.mean(-1)
                loss_list.append(loss_this_scale)
                acc_list.append(acc_this_scale)
                global_token_ptr = global_token_ptr + all_tokens
                global_scale_ptr += 1
        loss_list = torch.cat(loss_list)
        acc_list = torch.cat(acc_list)
        logits_norm = torch.mean(torch.tensor(logits_norm))
        return logits_norm, loss_list, acc_list
    
    def get_logits_during_infer(self, x_BLC, e):
        logits = self.head(x_BLC.float(), e)
        return logits
    
    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC: torch.Tensor,
        visual_rope_cache = None,
        sequece_packing_scales = None, # [[(1,1,1)->(5,5,5)], [(1,1,1)->(10,10,10)]] 1LC
        super_scale_lengths = None,
        other_info_by_scale = None,
        gt_BL = None,
        x_BLC_mask=None,
        scale_or_time_ids=None,
        return_last_hidden_states=False,
        use_chunk_id = -1,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        
        B = 1 # sequence packing
        cond_BD_or_gss, ca_kv = None, None
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
            x_BLC = chunk(x=x_BLC, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        logits_norm, loss_list, acc_list = self.get_loss_acc(x_BLC, x_BLC_mask, e, sequece_packing_scales, gt_BL, other_info_by_scale, return_last_hidden_states)
        return logits_norm, loss_list, acc_list, valid_sequence_ratio

    def prepare_text_conditions(
        self,
        label_B_or_BLT,
        negative_label_B_or_BLT,
        use_cfg=False,
    ):
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if use_cfg:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, lens
    
    @torch.no_grad()
    def autoregressive_infer(
        self,
        args=None,
        **kwargs,
    ):
        if 'infinity_refine' in args.dynamic_scale_schedule:
            infer_func = self.ar_infer_infinity_refine
        else:
            infer_func = self.autoregressive_infer_cfg
        return infer_func(args=args, **kwargs)

    def embeds_codes2input(
        self,
        last_stage, # [B, d, t, h, w]
    ):
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0,2,1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(last_stage) # norm0_ve is Identity
        return last_stage
    
    @torch.no_grad()
    def ar_infer_infinity_refine(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        noise_list=None,
        class_token_id=0,
        uncond_class_token_id=1000,
        reference_image_labels=None,
        **kwargs,
    ):   # returns List[idx_Bl]
        print(f'{class_token_id=} {uncond_class_token_id=}')
        from infinity.schedules.infinity_refine import shift_pt
        rng = None
        # if g_seed is None: rng = None
        # else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        text_scales = len(label_B_or_BLT) 
        each_round_scales = np.array(json.loads(args.each_round_scales))
        pbar = tqdm.tqdm(total=each_round_scales[0])
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks
        use_cfg = True
        cfg_interval = float(args.cfg_type.split('_')[-1])
        full_pt, ph, pw = scale_schedule[0]
        if args.first_frame_condition:
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
        
        magic_df_round = len(each_round_scales) # 3
        if args.refine_mode in ['ar_discrete_GRN_bit']:
            classes = 2
            labels_shape = (1,args.detail_scale_dim*magic_df_round,pt,ph,pw)
        elif args.refine_mode in ['ar_discrete_GRN_index']:
            classes = 2**magic_df_round
            labels_shape = (1,args.detail_scale_dim,pt,ph,pw)

        mul_pt_ph_pw = pt * ph * pw
        repeat_idx = -1
        scale_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+args.add_scale_token]
        if noise_list is not None:
            noise_list[0] = noise_list[0].to('cuda')
        assert len(scale_schedule) == 1
        if args.first_frame_condition:
            first_frame_labels = noise_list[0][:,:,:1] # [B,d,1,h,w]
            first_frame_tokens_cond = self.embeds_codes2input(multiclass_labels2onehot_input(first_frame_labels, classes))
            fist_frame_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (1, ph, pw), device, args.mapped_h_div_w_template, t_offset=0)
            visual_rope_cache = torch.cat((visual_rope_cache, fist_frame_rope_cache), dim=4)
            tmp_seqlens = [mul_pt_ph_pw + ph * pw + lens[0] + args.add_scale_token, mul_pt_ph_pw + lens[1] + ph * pw + args.add_scale_token]
        else:
            absolute_gt_labels = noise_list[0].permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
            tmp_seqlens = [mul_pt_ph_pw+lens[0]+args.add_scale_token, mul_pt_ph_pw+lens[1]+args.add_scale_token]
        
        # [visual tokens, text tokens, pt tokens]
        rope_cache = torch.cat([visual_rope_cache, rope_cache_text_cond, scale_token_rope_cache, visual_rope_cache, rope_cache_text_uncond, scale_token_rope_cache], dim=4) # (2, 1, 1, 1, seq_len, dim / 2)
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, dim / 2) -> (2, 1, 1, seq_len, dim / 2) -> (2, 1, seq_len, 1, dim / 2)

        cu_seqlens = torch.tensor([0]+tmp_seqlens, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(tmp_seqlens)
        speed_test = False
        for round_ind in range(1):
            cur_round_scales = each_round_scales[round_ind]
            pure_rand_labels = torch.randint(low=0, high=classes, size=labels_shape, device=infer_device, dtype=infer_dtype)
            mixed_xt = pure_rand_labels
            if args.min_infer_steps > 0:
                min_infer_steps, max_infer_steps = args.min_infer_steps, cur_round_scales
            else:
                min_infer_steps, max_infer_steps = cur_round_scales, cur_round_scales
            next_pt = 0.
            decision_entrophy = None
            for cur_inner_round_si in range(cur_round_scales):
                cur_pt = next_pt
                is_last_step = np.abs(cur_pt - 1) < 0.02
                if round_ind == 0 and cur_inner_round_si == 0:
                    self.entrophy_statistics.append([])
                repeat_idx += 1 # index scale tokens, very important
                cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0
                last_stage = self.embeds_codes2input(multiclass_labels2onehot_input(mixed_xt, classes)) # [B,d*num_classes,t,h,w]
                pt_tokens = self.pt_embedder(torch.tensor([cur_pt], device=device)).unsqueeze(0)
                # [visual tokens, text tokens, pt tokens]
                if args.first_frame_condition:
                    last_stage_cond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                    last_stage_uncond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
                else:
                    last_stage_cond = torch.cat((last_stage, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                    last_stage_uncond = torch.cat((last_stage, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
                    
                last_stage = torch.cat([last_stage_cond, last_stage_uncond], dim=1)
                e, e0 = None, None
                last_diffusion_step = False
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(x=last_stage, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask, attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache, last_diffusion_step=last_diffusion_step, use_cfg=use_cfg)
                logits = self.get_logits_during_infer(last_stage, e=e)
                tmp_bs, tmp_seq_len = logits.shape[:2]
                logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl) # [B,thw+...,d,2]
                pred_cond_logits = logits[:,:mul_pt_ph_pw] # [B,thw,d,2]
                pred_uncond_logits = logits[:,tmp_seqlens[0]:tmp_seqlens[0]+mul_pt_ph_pw] # [B,thw,d,2]
                
                if not speed_test:
                    pred_cond_probs = pred_cond_logits.softmax(-1) # [B,thw,d,2]
                    pred_cond_prob_zero = pred_cond_probs[...,0].mean()
                    pred_cond_prob_one = pred_cond_probs[...,1].mean()
                    categories = pred_cond_logits.shape[-1]
                    entrophy = (-pred_cond_probs * torch.log2(pred_cond_probs)).sum(-1).mean().item() / np.log2(categories)

                    decision_steps = 5
                    if cur_inner_round_si == decision_steps:
                        decision_entrophy = entrophy
                
                if args.sampling_method == 'shift':
                    pt_unshift = (cur_inner_round_si + 1) / (max_infer_steps - 1)
                    next_pt = shift_pt(pt_unshift, args.alpha)
                    next_pt = next_pt * 0.95
                elif args.sampling_method == 'cosine':
                    pt_unshift = (cur_inner_round_si + 1) / (max_infer_steps - 1)
                    pt_shift = shift_pt(min(1., pt_unshift), args.alpha)
                    next_pt = 1 - np.cos(np.pi/2*pt_shift)
                    next_pt = next_pt * 0.90
                else:
                    raise ValueError(f'{args.sampling_method=} is not supported')

                if not speed_test:
                    pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1) # [B,thw,d]
                    pred_cond_labels = Bld2Bthwd(pred_cond_labels, pt, ph, pw)

                cur_cfg = cfg # if cur_pt < 0.2 else 2.
                if cur_cfg != 1:
                    pred_cfg_logits = pred_uncond_logits + cur_cfg * (pred_cond_logits - pred_uncond_logits)
                else:
                    pred_cfg_logits = pred_cond_logits
                cur_tau = tau_list[0]
                pred_cfg_logits = pred_cfg_logits / cur_tau # [B,thw,d,2]
                pred_cfg_probs = pred_cfg_logits.softmax(dim=-1) # [B,thw,d,2]
                
                if not speed_test:
                    pred_cfg_prob_zero = pred_cfg_probs[...,0].mean()
                    pred_cfg_prob_one = pred_cfg_probs[...,1].mean()
                    pred_cfg_labels = torch.argmax(pred_cfg_probs, dim=-1) # [B,thw,d]
                    pred_cfg_labels = Bld2Bthwd(pred_cfg_labels, pt, ph, pw) # [B,t,h,w,d]
                pred_sample_labels = torch.multinomial(pred_cfg_probs.view(-1, args.detail_num_lvl), num_samples=1, replacement=True, generator=rng).view(tmp_bs, mul_pt_ph_pw, -1) # [B, thw,d]
                if not speed_test:
                    pred_sample_probs = torch.gather(pred_cfg_probs, dim=3, index=pred_sample_labels.unsqueeze(-1)).squeeze(-1) # [B,thw,d]
                    pred_sample_probs = Bld2Bthwd(pred_sample_probs, pt, ph, pw) # [B,t,h,w,d]
                pred_sample_labels = Bld2Bthwd(pred_sample_labels, pt, ph, pw) # [B,t,h,w,d]

                if not speed_test:
                    assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100. # different ratio between prediciton and input
                    pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.
                    pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.
                    mixed_xt_Bthwd_01 = mixed_xt.clone().permute(0,2,3,4,1)
                    mixed_xt_Bthwd_01[mixed_xt_Bthwd_01<0] = 0
                    pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.
                    pred_cfg_flip_ratio = (pred_cfg_labels != mixed_xt_Bthwd_01).sum() / pred_cfg_labels.numel() * 100.
                    pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.
                    self.entrophy_statistics[-1].append({
                        'round_ind': round_ind,
                        'cur_inner_round_si': cur_inner_round_si,
                        'cur_pt': cur_pt,
                        'cur_tau': cur_tau,
                        'cur_cfg': cur_cfg,
                        'entrophy': entrophy,
                        'assume_flip_ratio': assume_flip_ratio,
                        'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                        'pred_cfg_flip_ratio': pred_cfg_flip_ratio.item(),
                        'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
                        'uncond_class_token_id': uncond_class_token_id,
                        'pred_zero_ratio': pred_zero_ratio.item(),
                        'pred_one_ratio': pred_one_ratio.item(),
                        'pred_cond_prob_zero': pred_cond_prob_zero.item(),
                        'pred_cond_prob_one': pred_cond_prob_one.item(),
                        'pred_cfg_prob_zero': pred_cfg_prob_zero.item(),
                        'pred_cfg_prob_one': pred_cfg_prob_one.item(),
                        'meta': args.meta,
                    })
                    print(f'{cur_inner_round_si=} {cur_pt=:.3f} {cur_tau=:.3f} {cur_cfg=} {pred_sample_labels.shape=}')
                    print(f'{assume_flip_ratio=:.2f}% {pred_cond_flip_ratio=:.2f}% {pred_cfg_flip_ratio=:.2f}% {pred_sample_flip_ratio=:.2f}%')
                if cur_pt < gt_leak:
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
                # pred_sample_probs = pred_sample_probs.permute(0,4,1,2,3) # [B,t,h,w,d] -> [B,d,t,h,w]
                use_predict_mask = torch.rand(pred_sample_labels.shape, device=device) < next_pt
                if args.resample_rand_labels_per_step == 1:
                    cur_rand_labels = torch.randint(low=0, high=classes, size=pred_sample_labels.shape, device=pred_sample_labels.device, dtype=pred_sample_labels.dtype)
                    mixed_xt = torch.where(use_predict_mask, pred_sample_labels, cur_rand_labels)
                else:
                    mixed_xt = torch.where(use_predict_mask, pred_sample_labels, pure_rand_labels)
                next_pt = use_predict_mask.float().mean().item() # 0~1, list
                pbar.update(1)
                
                if args.save_each_step_result:
                    from infinity.models.hbq import bit_label2raw_feature
                    approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=magic_df_round) # [B, hbq_round_mul_d, t, h, w] -> [B,d,t,h,w]
                    generated_image = self.summed_codes2images(vae, approx_signal)
                    from tools.run_infinity import images2video
                    save_file = osp.join(args.save_dir, f'details/{cur_inner_round_si:03d}_{cur_pt:.3f}.mp4')
                    os.makedirs(osp.dirname(save_file), exist_ok=True)
                    images2video(generated_image[0].cpu().numpy(), fps=args.fps, save_filepath=save_file)
                
                if is_last_step: break

        if args.first_frame_condition:
            pred_sample_labels = torch.cat((first_frame_labels, pred_sample_labels), dim=2)
        if args.refine_mode == 'ar_discrete_GRN_index':
            from infinity.models.hbq import index_label2quant_features
            approx_signal = index_label2quant_features(pred_sample_labels, hbq_round=magic_df_round)
        elif args.refine_mode == 'ar_discrete_GRN_bit':
            from infinity.models.hbq import bit_label2raw_feature
            approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=magic_df_round) # [B, hbq_round_mul_d, t, h, w] -> [B,d,t,h,w]
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        img = self.summed_codes2images(vae, approx_signal)
        img_256 = img
        return ret, idx_Bl_list, img, img_256

    @torch.no_grad()
    def scale_or_time_embeds_infer(
        self,
        scale_or_time_id,
        length,
        device,
    ):
        scale_or_time_ids = torch.repeat_interleave(
            torch.tensor([scale_or_time_id], device=device),
            torch.tensor([length], device=device), 
        )
        scale_or_time_ids = scale_or_time_ids.unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.scale_or_time_embedding(sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()) # [1, visual_seq_len,] -> [1, visual_seq_len, 256] -> [1, visual_seq_len, C]
            e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.C)) # [1, L, C] -> [1, L, 6C] -> [1, L, 6, C]
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        return e, e0

    def summed_codes2images(self, vae, summed_codes):
        t1 = time.time()
        if self.task_type == 't2i':
            img = vae.decode(summed_codes.squeeze(-3))
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            img = img.permute(0, 2, 3, 1) # [bs, 3, h, w] -> [bs, h, w, 3]
            img = img.mul_(255).to(torch.uint8).flip(dims=(3,))
        else: # 't2v'
            img = vae.decode(summed_codes, slice=True)
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            img = img.permute(0,2,3,4,1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
            img = img.mul_(255).to(torch.uint8).flip(dims=(4,))
        print(f'Decode takes {time.time()-t1:.1f}s')
        return img # bgr order

    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):       
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        if self.arch == 'qwen':
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
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError

TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa2b(depth=28, block_chunks=7, embed_dim=2560, num_heads=2560//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa8b(depth=42, block_chunks=7, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa14b(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, drop_path_rate=0.1, mlp_ratio=3.4, **kwargs): 
    return Infinity(
        depth=depth, 
        block_chunks=block_chunks, 
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        mlp_ratio=mlp_ratio, 
        drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
    # (depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128//4, drop_path_rate=0, **kwargs)

@register_model
def infinity_sa12b(depth=60, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_sa16b(depth=42, embed_dim=4096, num_heads=4096//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_v2b(depth=32, embed_dim=2016, num_heads=2016//126, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_8b(depth=40, block_chunks=1, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, **kwargs): return Infinity(depth=depth, block_chunks=block_chunks, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def infinity_qwen7b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=12288/4096, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
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
def infinity_qwen2_2b(depth=28, block_chunks=7, embed_dim=2304, num_heads=2304//128, num_key_value_heads=2304//128, drop_path_rate=0, **kwargs): 
    return Infinity(
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
def infinity_qwen8b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, mlp_ratio=4, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
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
def infinity_qwen8b_v2(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128, mlp_ratio=12288/4096, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
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
def infinity_qwen_wide14b(depth=36, block_chunks=6, embed_dim=5632, num_heads=5632//128, num_key_value_heads=5632//128//4, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen13bMHA(depth=40, block_chunks=8, embed_dim=5120, num_heads=5120//128, num_key_value_heads=5120//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=True,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.4,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def dit_xl_4step(depth=28, block_chunks=4, embed_dim=1152, num_heads=1152//128, num_key_value_heads=1152//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def dit_xl_28step(depth=28, block_chunks=28, embed_dim=1152, num_heads=1152//128, num_key_value_heads=1152//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def dit_xl_1step(depth=28, block_chunks=1, embed_dim=1152, num_heads=1152//128, num_key_value_heads=1152//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def dit_xl_1block1step(depth=1, block_chunks=1, embed_dim=1152, num_heads=1152//128, num_key_value_heads=1152//128, drop_path_rate=0, **kwargs): 
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4.0,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_qwen_0b_4step(depth=4, block_chunks=4, embed_dim=512, num_heads=512//128, num_key_value_heads=512//128, drop_path_rate=0, **kwargs): 
    return Infinity(
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
def infinity_qwen0b(depth=4, block_chunks=2, embed_dim=512, num_heads=512//128, num_key_value_heads=512//128, drop_path_rate=0, **kwargs): 
    return Infinity(
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
def infinity_qwen2_30b(depth=54, block_chunks=27, embed_dim=6144, num_heads=6144//128, num_key_value_heads=6144//128//4, drop_path_rate=0, **kwargs):
    return Infinity(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=4, #mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def infinity_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def infinity_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def infinity_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return Infinity(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
