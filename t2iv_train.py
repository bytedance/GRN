import gc
import json
import math
import os
import os.path as osp
import random
import sys
import time
from functools import partial
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import grn.utils_t2iv.dist as dist
from grn.dataset.build import build_joint_dataset
from grn.models.ema import get_ema_model
from grn.utils_t2iv import arg_util, misc
from grn.utils import wandb_utils
from grn.trainer import get_trainer

def build_everything_from_args(args: arg_util.Args, saver):
    args.set_initial_seed(benchmark=True)
    print(f'Loading T5 from {args.t5_path}...')
    from grn.models.umt5.t5 import T5EncoderModel
    text_encoder = T5EncoderModel(
        text_len=args.tlen, # 512
        dtype=torch.bfloat16, # torch.bfloat16
        device=args.device,
        checkpoint_path=osp.join(args.t5_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=osp.join(args.t5_path, 'umt5-xxl'),
        enable_fsdp=True) # False
    # text_encoder.model.to(args.device)
    text_tokenizer = text_encoder.tokenizer
    args.text_tokenizer_type = 'umt5'
    args.text_tokenizer = text_tokenizer

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args)
    
    Trainer = get_trainer(args)
    # build trainer
    trainer = Trainer(
        is_visualizer=dist.is_visualizer(), device=args.device, 
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp,
        zero=args.zero, vae_latent_dim=args.vae_latent_dim, gpt_opt=gpt_optim,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,
    )
    
    # auto resume from broken experiment
    global_it = 0
    if args.checkpoint_type == 'torch':
        from grn.utils_t2iv.save_and_load import auto_resume
        auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
        print(f'initial args:\n{str(args)}')
        if start_ep == args.ep:
            print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
            return None
        if trainer_state is not None and len(trainer_state):
            trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
        
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return text_tokenizer, text_encoder, trainer, global_it


def build_model_optimizer(args):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from grn.models.grn import MultipleLayers
    from grn.models.init_param import init_weights
    from grn.utils_t2iv.lr_control import filter_params
    from grn.utils_t2iv.load import build_vae_gpt
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    if num_para/1000 < 20: # < 20B
        gpt_wo_ddp = gpt_wo_ddp.to('cuda')

    init_weights(gpt_wo_ddp)
    gpt_wo_ddp.special_init()
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    
    if args.rush_resume:
        print(f"{args.rush_resume=}")
        if '.pth' in args.rush_resume:
            cpu_d = torch.load(args.rush_resume, 'cpu')
        else:
            from grn.utils_t2iv.save_and_load import merge_ckpt
            cpu_d = merge_ckpt(args.rush_resume, osp.join(args.rush_resume, 'ouput'), save=False, use_ema_model=False, fsdp_save_flatten_model=args.fsdp_save_flatten_model)
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            try:
                if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                    print(f'[rush_resume] drop word_embed.weight')
                    del state_dict['word_embed.weight']
                if 'head.proj.weight' in state_dict and (state_dict['head.proj.weight'].shape[0] != gpt_wo_ddp.head.proj.out_features):
                    print(f'[rush_resume] drop head')
                    del state_dict['head.proj.weight']
                    del state_dict['head.proj.bias']
            except Exception as e:
                print(e)
                for key in ['word_embed.weight', 'head.proj.weight', 'head.proj.bias']:
                    if key in state_dict:
                        del state_dict[key]
                        print(f'[rush_resume] drop {key}')

            if 'text_proj_for_sos.ca.mat_kv.weight' in state_dict and \
                (state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape):
                print(f'[rush_resume] drop cfg_uncond')
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        print(gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False))
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    print(f'[PT] GPT model details:')
    for name, param in gpt_wo_ddp.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")
    print(f'[PT][#para], GPT={num_para:.2f}M parameters\n\n')
    
    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh
        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if args.fsdp_warp_mode == 'full':
            print(f'warp all modules for fsdp')
            def my_policy(
                module: torch.nn.Module,
                recurse: bool,
                **kwargs,
            ) -> bool:
                return True
            auto_wrap_policy = my_policy
        else:
            print(f'warp transformer blocks for fsdp')
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard == 1:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree <= world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        elif args.enable_hybrid_shard == -1: # no shard
            sharding_strategy = ShardingStrategy.NO_SHARD
            device_mesh = None
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)

        if args.fsdp_init_device == 'cpu':
            gpt_wo_ddp = gpt_wo_ddp.cpu()

        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=True, 
                sync_module_states=True, 
                limit_all_gathers=True,
                device_mesh=device_mesh,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    nowd_keys |= {
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        'text_proj_for_sos.ca.mat_q',
        'scale_tokens', 'class_tokens'
    }
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    opt_clz = partial(torch.optim.AdamW, betas=(0.9, 0.999), fused=True)
    opt_kw = dict(lr=args.tlr, weight_decay=args.twd)
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = opt_clz(params=para_groups, **opt_kw)
    del names, paras, para_groups
    return vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim

def build_dataset(args):
    train_dataset = build_joint_dataset(
        args, 
        args.meta_folders,
        args.meta_folder_repeats,
        max_caption_len=args.tlen, 
        short_prob=args.short_cap_prob, 
        load_vae_instead_of_image=False
    )
    return train_dataset

def main_train(args: arg_util.Args):
    if args.checkpoint_type == 'torch':
        from grn.utils_t2iv.save_and_load import CKPTSaver, auto_resume
        saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    else:
        raise ValueError(f'{args.checkpoint_type=}')
    text_tokenizer, text_encoder, trainer, start_global_it = build_everything_from_args(args, saver)
    gc.collect(), torch.cuda.empty_cache()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    # ============================================= epoch loop begins =============================================
    # build wandb logger
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed + tdist.get_rank())
    for ep in range(args.ep):
        # build data at each epoch to ensure read meta take effects for each dataloader worker
        args.epoch = ep
        train_dataset = build_dataset(args)
        iters_train = len(train_dataset)
        print(f'[PT info]  from {start_global_it=} {iters_train=}=======>  bed: {args.bed}  <=======\n')
        
        # build dataloader
        train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.workers, pin_memory=True, batch_size=None, shuffle=True, generator=dataloader_generator)
        train_dataloader_iter_obj = iter(train_dataloader)

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            start_global_it=start_global_it,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=train_dataloader_iter_obj,
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
        )
        start_global_it += iters_train
        del stats, train_dataset, train_dataloader
        time.sleep(10), gc.collect(), time.sleep(10) # torch.cuda.empty_cache()
    return


def train_one_ep(
    ep: int, start_global_it: int, me: misc.MetricLogger,
    saver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    g_it, max_it = start_global_it, args.ep * iters_train
    
    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['L', 'L_i', 'L_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Acc', 'Acc_i', 'Acc_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['seq_usage']]
    # ============================================= iteration loop begins =============================================
    start_it = 0
    for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
        # for dpo training, we will save the first iter model for comparison
        g_it += 1
        if (g_it > 0 and g_it % args.save_model_iters_freq == 0) or (args.save_start_model and g_it == 1):
            if args.checkpoint_type == 'torch':
                saver.sav(args=args, g_it=g_it, next_ep=ep, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
        
        # [get data]
        images, captions, raw_features_bcthw, feature_cache_files4images, media, meta_list = data['images'], data['captions'], data['raw_features_bcthw'], data['feature_cache_files4images'], data['media'], data['meta_list']

        # # [prepare text features]
        if args.add_class_token > 0: # c2i task
            text_cond_tuple = [[] for _ in range(5)]
        else:
            caption_nums = [len(item) for item in captions]
            flatten_captions = []
            for item in captions:
                flatten_captions.extend(item)
            if args.text_tokenizer_type == 'flan_t5':
                tokens = text_tokenizer(text=flatten_captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                kv_compact = []
                for text_ind, (len_i, feat_i) in enumerate(zip(lens, text_features.unbind(0))):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext, caption_nums)
            else:
                text_features = text_encoder(flatten_captions, args.device)
                lens = [len(item) for item in text_features]
                cu_seqlens_k = [0]
                for len_i in lens:
                    cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
                cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
                Ltext = max(lens)
                kv_compact = torch.cat(text_features, dim=0).float()
                text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext, caption_nums)

        if len(images):
            images = [item.to(args.device, non_blocking=True) for item in images]
        if len(raw_features_bcthw):
            raw_features_bcthw = [item.to(args.device, non_blocking=True) for item in raw_features_bcthw]
        
        # [schedule learning rate and weight decay]
        if ep == 0 and (g_it-start_global_it) < args.wp_it:
            cur_lr_ratio = args.wp0 + (1-args.wp0) * (g_it-start_global_it) / args.wp_it
        else:
            cur_lr_ratio = 1
        cur_lr = args.tlr * cur_lr_ratio
        if cur_lr_ratio < 1:
            cur_wd = args.twd
            for param_group in trainer.gpt_opt.param_groups:
                param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
                param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
        
        # [get scheduled hyperparameters]
        stepping = (g_it + 1) % args.gradient_accumulation == 0
        step_cnt += int(stepping)
        
        trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=1,
            metric_lg=me, 
            logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
            inp_B3HW=images, 
            raw_features_bcthw=raw_features_bcthw,
            feature_cache_files4images=feature_cache_files4images,
            text_cond_tuple=text_cond_tuple,
            media=media,
            meta_list=meta_list,
            args=args,
        )
        
        me.update(tlr=cur_lr)
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


def main():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main_train(args)
    print(f'final args:\n\n{str(args)}')
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()
    time.sleep(120)


if __name__ == '__main__':
    main()
