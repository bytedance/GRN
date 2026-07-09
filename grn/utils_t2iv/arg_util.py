import json
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict, deque
from typing import Optional, Union, Literal

import numpy as np
import torch
from tap import Tap

import grn.utils_t2iv.dist as dist
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager


class Args(Tap):
    local_out_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_output')  # directory for save checkpoints
    data_path: str = ''                 # dataset path
    video_fps: int = 24                 # video fps
    video_frames: int = 1               # video frames
    hdfs_mode: str = 'read'             # hdfs_mode
    bed: str = ''                       # bed directory for copy checkpoints apart from local_out_path
    vae_path: str = ''                  # VAE ckpt
    exp_name: str = ''                  # experiment name
    model: str = ''                     # for VAE training, 'b' or any other for GPT training
    short_cap_prob: float = 0.2         # prob for training with short captions
    project_name: str = ''              # name of wandb project
    tf32: bool = True                   # whether to use TensorFloat32
    auto_resume: bool = True            # whether to automatically resume from the last checkpoint found in args.bed
    rush_resume: str = ''               # pretrained checkpoint
    enable_hybrid_shard: int = 0        # whether to use hybrid FSDP
    inner_shard_degree: int = 1         # inner degree for FSDP
    zero: int = 0                       # ds zero
    enable_checkpointing: str = None    # checkpointing strategy: full-block, self-attn
    pad_to_multiplier: int = 1          # >1 for padding the seq len to a multiplier of this
    log_every_iter: bool = False
    checkpoint_type: str = 'torch'      # checkpoint_type: torch, onmistore
    device: str = 'cpu'
    is_master_node: bool = None
    # dir
    log_txt_path: str = ''
    t5_path: str = ''                   
    online_t5: bool = True              # whether to use online t5 or load local features
    # GPT
    sdpa_mem: bool = True               # whether to use with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    tfast: int = 0                      # compile GPT
    tau: float = 1                      # tau of self attention in GPT
    tp: float = 0.0                     # top-p
    tk: float = 0.0                     # top-k
    drop_condition_prob: float = 0.1    # >0: classifier-free guidance, drop cond with prob cfg
    fp16: int = 0                       # 1: fp16, 2: bf16, >2: fp16's max scaling multiplier todo: 记得让quantize相关的feature都强制fp32！另外residueal最好也是fp32（根据flash-attention）nn.Conv2d有一个参数是use_float16？
    use_flex_attn: bool = False         # whether to use flex_attn to speedup training
    tlr: float = 2e-5                   # learning rate
    twd: float = 0.005                  # vqgan: 0.01
    twde: float = 0
    ep: int = 100
    wp: float = 0
    wp0: float = 0.005
    wpe: float = 0.3                    # 0.001, final cosine lr = wpe * peak lr
    sche: str = ''                      # cos, exp, lin
    log_freq: int = 50                  # log frequency in the stdout
    tclip: float = 2.                   # <=0 for not grad clip GPT; >100 for per-param clip (%= 100 automatically)
    # data
    workers: int = 0                    # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    norm_eps: float = 1e-6              # norm eps
    tlen: int = 512                     # truncate text embedding to this length
    Ct5: int = 2048                     # feature dimension of text encoder
    num_of_label_value: int = 2         # num_of_label_value, =2 means bitwise label, =0 means index-wise label, others means fsq, never set to 1
    enable_dynamic_length_prompt: int = 0 # enable dynamic length prompt during training
    save_model_iters_freq: int = 1000   # save model iter freq
    reweight_loss_by_scale: float = 0     # reweight loss by scale
    vae_latent_dim: int = 1                   # here 16/32/64 is bsq vae of different quant bits
    model_init_device: str = 'cuda'     # model_init_device
    fsdp_init_device: str = 'cuda'     # model_init_device
    apply_spatial_patchify: int = 0     # apply apply_spatial_patchify or not
    dynamic_scale_schedule: str = ''    # dynamic scale schedule
    use_slice: int = 0                  # whether use slice for vae encoding
    use_vae_token_cache: int = 1        # whether use token cache for speedup
    save_vae_token_cache: int = 0       # whether save_vae_token_cache
    allow_online_vae_feature_extraction: int = 1 # whether allow_online_vae_feature_extraction, if False, only load cached features
    use_text_token_cache: int = 1       # whether use text token cache for speedup
    token_cache_dir: str = ''           # token_cache_dir
    down_size_limit: int = 10000        # down_size_limit in MB, larger video won't download
    addition_pn_list: str = '[]'
    video_caption_type: str = 'merged_caption'
    video_caption_type: str = ''        # video caption type, we use tarsier2_caption
    only_images4extract_feats: int = 0  # only extract feats for images, set true when extract features, set false for training
    temporal_compress_rate: int = 4     # temporal_compress_rate, set to 4 by default
    cached_video_frames: int = 81       # load cache files' video_frames, set to 81 by default
    rope_type: str = '3d'               # rope type, choose from ['2d', '3d', '4d'], default set to '3d'
    loop_data_per_epoch: int = 0
    meta_folders: str = ''
    meta_folder_repeats: str = ''
    meta_folder_identifiers: str = ''
    pn_list: str = ''
    pn_probs: str = ''
    i2v_ratio: float = 0.
    dense_ratio4seqpack: float = 1.
        
    # RL Arguments
    pair_input: int = 0 # dpo needs pair_input=1
    rl_with_ref_model: int = 1
    dpo_training: int = 0 # whether enable dpo training
    dpo_loss_type: str = 'dpo'
    dpo_beta: float = 1.0 # 0.1~0.5, larger dpo_beta will be more close to reference model
    dpo_func: str = 'sigmoid'
    dpo_label_smoothing: float = 0.0
    dpo_sft_weight: float = 1.0 # use sft_weight when doing dpo
    scale_wise_dpo: int = 0

    # ema model or rl reference model
    use_fsdp_model_ema: int = 0
    model_ema_decay: float = 0.9999 # model_ema_decay < 1 will update the ema model, >=1 will fix the model and is used as rl reference model

    # seq parallel
    sp_size: int = 0

    train_max_token_len: int = -1
    duration_resolution: float = 1
    cache_check_mode: int = 0 # 0 means not check chche file, 1 means check at the begining, 2 means check at each iteration, -1 means include no cache meta only, used for token cache
    wp_it: int = 100
    drop_long_video: int = 1

    image_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
    video_scale_repetition: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
    video_scale_probs: str = '[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.2, 0.1, 0.05]'
    hbq_round: int = 4
    train_h_div_w_list: str = '[]'
    simple_text_proj: int = 0
    min_video_frames: int = -1
    fsdp_save_flatten_model: int = 0
    semantic_scale_dim: int = 16
    detail_scale_dim: int = 64
    restrict_data_size: int = -1
    skip_count_text_token: int = 0
    semantic_num_lvl: int = 2
    detail_num_lvl: int = 2
    use_fsq_cls_head: Literal[0, 1] = 0
    use_clipwise_caption: int = 0
    fsdp_warp_mode: str = 'trans_block' # trans_block or full
    save_start_model: int = 0
    use_ada_layer_norm: int = 0
    add_scale_token: Literal[0, 1] = 0 # must >= 0
    add_class_token: int = 0 # must >= 0, > 0 means class2image
    vae_encoder_out_type: str = 'mu_sigma'
    alpha: float = 0.0
    refine_mode: str = ''
    log_norm_mean: float = 0
    log_norm_sigma: float = -1. # < 0 mean disable log-norm sampling
    gradient_accumulation: int = 1
    # would be automatically set in runtime
    cmd: str = ' '.join(a.replace('--exp_name=', '').replace('--exp_name ', '') for a in sys.argv[7:])  # [automatically set; don't specify this]
    
    @property
    def gpt_training(self):
        return len(self.model) > 0

    def set_initial_seed(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        assert self.seed
        seed = self.seed
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def compile_model(self, m, fast):
        if fast == 0:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m
    
    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        # self.as_dict() would contain methods, but we only need variables
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                d[k] = getattr(self, k)
        return d
    
    @staticmethod
    def set_tf32(tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')
                print(f'[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}')
            print(f'[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}')
            print(f'[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
    
    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {'device', 'dbg_ks_fp'}:     # these are not serializable
                s.append(f'  {k:20s}: {getattr(self, k)}')
        s = '\n'.join(s)
        return f'{{\n{s}\n}}\n'


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--local-rank=') or sys.argv[i].startswith('--local_rank='):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    args.chunk_nodes = int(os.environ.get('CK', '') or '0')
    
    if len(args.extra_args) > 0 and args.is_master_node == 0:
        print(f'======================================================================================')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================\n{args.extra_args}')
        print(f'=========================== WARNING: UNEXPECTED EXTRA ARGS ===========================')
        print(f'======================================================================================\n\n')
    
    args.set_tf32(args.tf32)
    
    try: os.makedirs(args.bed, exist_ok=True)
    except: pass
    try: os.makedirs(args.local_out_path, exist_ok=True)
    except: pass
    
    day3 = 60*24*3
    dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=day3 if int(os.environ.get('LONG_DBG', '0') or '0') > 0 else 30)
    args.device = dist.get_device()

    # sync seed
    args.seed = int(time.time())
    seed = torch.tensor([args.seed], device=args.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(seed, op=torch.distributed.ReduceOp.MIN)
    args.seed = seed.item()

    if args.sp_size > 1:
        print(f"INFO: sp_size={args.sp_size}")
        sp_manager.init_sp(args.sp_size)
    
    args.sche = args.sche or ('lin0' if args.gpt_training else 'cos')
    if args.wp == 0:
        args.wp = args.ep * 1/100
    
    di = {
        'b': 'bilinear', 'c': 'bicubic', 'n': 'nearest', 'a': 'area', 'aa': 'area+area',
        'at': 'auto', 'auto': 'auto',
        'v': 'vae',
        'x': 'pix', 'xg': 'pix_glu', 'gx': 'pix_glu', 'g': 'pix_glu'
    }
    
    args.log_txt_path = os.path.join(args.local_out_path, 'log.txt')
    args.video_scale_probs = json.loads(args.video_scale_probs)
    
    ls = '[]'
    if 'AUTO_RESUME' in os.environ:
        ls.append(int(os.environ['AUTO_RESUME']))
    ls = sorted(ls, reverse=True)
    ls = [str(i) for i in ls]
    args.ckpt_trials = ls
    args.real_trial_id = args.trial_id if len(ls) == 0 else str(ls[-1])
    
    args.enable_checkpointing = None if args.enable_checkpointing in [False, 0, "0"] else args.enable_checkpointing
    args.enable_checkpointing = "full-block" if args.enable_checkpointing in [True, 1, "1"] else args.enable_checkpointing
    assert args.enable_checkpointing in [None, "full-block", "full-attn", "self-attn"], \
        f"only support no-checkpointing or full-block/full-attn checkpointing, but got {args.enable_checkpointing}."
    
    if len(args.exp_name) == 0:
        args.exp_name = os.path.basename(args.bed) or 'test_exp'
    
    if dist.is_master():
        from grn.utils.safe_rm import safe_remove
        safe_remove(os.path.join(args.bed, "ready-node*"), args.bed)
        safe_remove(os.path.join(args.local_out_path, "ready-node*"), args.local_out_path)

    if args.sdpa_mem:
        from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(False)
    print(args)
    return args
