import gc
import os
import os.path as osp
import random
import sys
from copy import deepcopy
from typing import Tuple, Union

import torch
import yaml

from grn.models.grn import *
from grn.models.hbq_tokenizer import HBQ_Tokenizer
from timm.models import create_model

def load_visual_tokenizer(args, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    vae = HBQ_Tokenizer(args=args, latent_channels=args.detail_scale_dim, encoder_out_type='feature_tanh')
    vae.eval()
    vae = vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False
    state_dict = torch.load(args.vae_path, map_location=device)
    if 'ema' in state_dict:
        print(f'Load ema vae weights')
        state_dict = state_dict['ema']
    else:
        print(f'Load non ema vae weights')
        state_dict = state_dict['vae']
    print('Load vae: ', vae.load_state_dict(state_dict, assign=True))
    return vae

def build_vae_gpt(args, device='cuda'):
    vae_local = load_visual_tokenizer(args, device)
    gpt_kw = dict(
        pretrained=False, global_pool='',
        text_channels=args.Ct5, text_maxlen=args.tlen,
        norm_eps=args.norm_eps,
        top_p=args.tp, top_k=args.tk, tau=args.tau,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        num_of_label_value=args.num_of_label_value,
        train_h_div_w_list=None,
        apply_spatial_patchify=args.apply_spatial_patchify,
        dynamic_scale_schedule=args.dynamic_scale_schedule,
        video_frames=args.video_frames,
        other_args=args,
    )
    print(f'[create gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    gpt_wo_ddp = create_model(args.model, **gpt_kw)
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    return vae_local, gpt_wo_ddp
