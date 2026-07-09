import random
import time
import gc
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os
import os.path as osp
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2

import grn.utils_t2iv.dist as dist
from grn.models.ema import update_ema
from grn.utils_t2iv import arg_util, misc
from grn.utils import wandb_utils
from grn.schedules import get_encode_decode_func
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils.compress_tokens import save_packed_tensor
from grn.models.grn import GRN

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

def sum_dict(acc_pt2scale_acc):
    for full_pt in acc_pt2scale_acc:
        for si in range(len(acc_pt2scale_acc[full_pt])):
            acc_pt2scale_acc[full_pt][si] = torch.tensor(acc_pt2scale_acc[full_pt][si]).sum()
    return acc_pt2scale_acc

def dict2list(acc_pt2scale_acc):
    flatten_acc_pt2scale_acc = []
    for key, val in acc_pt2scale_acc.items():
        flatten_acc_pt2scale_acc.extend(val)
    return flatten_acc_pt2scale_acc

def list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc):
    ptr = 0
    for key in acc_pt2scale_acc:
        for ind in range(len(acc_pt2scale_acc[key])):
            acc_pt2scale_acc[key][ind] = flatten_acc_pt2scale_acc[ptr]
            ptr += 1
    return acc_pt2scale_acc

import queue
import threading

def save_token():
    while True:
        try:
            raw_features, feature_cache_files4images = save_token_queue.get()
            for i in range(len(feature_cache_files4images)):
                if not osp.exists(feature_cache_files4images[i]):
                    os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                    # torch.save(raw_features[i], feature_cache_files4images[i])
                    save_packed_tensor(feature_cache_files4images[i], raw_features[i])
                    print(f'Save to {feature_cache_files4images[i]}')
                else:
                    print(f'{feature_cache_files4images[i]} exists, skip')
        except Exception as e:
            print(f"Error saving token: {e}")
        finally:
            save_token_queue.task_done()

save_token_queue = queue.Queue()
saver = threading.Thread(target=save_token, daemon=True)
saver.start()

class Trainer(object):
    def __init__(
        self, is_visualizer: bool, device, 
        vae_local, gpt_wo_ddp: GRN, gpt: DDP,  gpt_opt: torch.optim.Optimizer, 
        dbg_unused=False,zero=0, vae_latent_dim=True, reweight_loss_by_scale=0,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None,
    ):
        super(Trainer, self).__init__()
        self.zero = zero
        self.vae_latent_dim = vae_latent_dim
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        self.dynamic_scale_schedule = other_args.dynamic_scale_schedule
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames)
        self.gpt_opt = gpt_opt
        self.gpt_wo_ddp = gpt_wo_ddp
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.reweight_loss_by_scale = reweight_loss_by_scale
        video_encode, _, _, _ = get_encode_decode_func(other_args.dynamic_scale_schedule)
        self.video_encode = video_encode
        self.is_visualizer = is_visualizer
        numpy_generator = np.random.default_rng(other_args.seed + tdist.get_rank())
        torch_cuda_generator = torch.Generator(device=other_args.device)
        torch_cuda_generator.manual_seed(other_args.seed + tdist.get_rank())
        self.rank_vary_generator = {
            'numpy_generator': numpy_generator,
            'torch_cuda_generator': torch_cuda_generator,
        }
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        raw_features_bcthw: FTen, feature_cache_files4images: list, media: str, meta_list: list,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        device = args.device
        B = len(inp_B3HW) + len(raw_features_bcthw)

        if media == 'images':
            is_image_batch = 1
        else:
            is_image_batch = 0
        # [forward]
        with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=args.zero == 0):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                raw_features_list = []
                if len(inp_B3HW):
                    with torch.no_grad():
                        for inp_ind, inp in enumerate(inp_B3HW):
                            raw_features_, _, _ = self.vae_local.encode_for_raw_features(inp.unsqueeze(0), scale_schedule=None, slice=args.use_slice)
                            raw_features = raw_features_[0]
                            save_tokens = args.use_vae_token_cache and args.save_vae_token_cache and (not osp.exists(feature_cache_files4images[inp_ind]))
                            if save_tokens:
                                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
                                saved_visual_features = raw_feature2bit_label(raw_features, args.hbq_round).to(torch.bool)
                                saved_visual_features = saved_visual_features.cpu().data
                                save_token_queue.put((saved_visual_features, [feature_cache_files4images[inp_ind]]))
                            raw_features_list.append([raw_features])
                                
                if len(raw_features_bcthw):
                    raw_features_bcthw = [[item.unsqueeze(0)] for item in raw_features_bcthw]
                    raw_features_list = raw_features_list + raw_features_bcthw

            assert isinstance(raw_features_list[0], list)
            full_pts_this_batch = [item[0].shape[-3] for item in raw_features_list]
            kv_compact, lens, cu_seqlens_k, max_seqlen_k, caption_nums = text_cond_tuple
            
            with torch.no_grad():
                x_BLC, x_BLC_mask, scale_or_time_ids, gt_BLC, _, visual_rope_cache, sequece_packing_scales, super_scale_lengths, other_info_by_scale = self.video_encode(
                    vae=self.vae_local,
                    inp_B3HW=None,
                    vae_features=raw_features_list,
                    args=args,
                    device=device,
                    rope2d_freqs_grid=self.gpt.rope2d_freqs_grid,
                    dynamic_resolution_h_w=self.dynamic_resolution_h_w,
                    text_lens=lens,
                    caption_nums=caption_nums,
                    tokens_remain=args.train_max_token_len,
                    rank_vary_generator=self.rank_vary_generator,
                    vis_verbose=False, # g_it%20==0,
                    meta_list=meta_list,
                )
            
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # import pdb; pdb.set_trace()
            # from torchinfo import summary
            # res = summary(self.gpt, input_data=(text_cond_tuple, x_BLC_wo_prefix, scale_schedule))

            logits_norm, loss, acc_bit, valid_sequence_ratio = self.gpt(
                text_cond_tuple,
                x_BLC,
                x_BLC_mask=x_BLC_mask,
                gt_BL=gt_BLC,
                is_image_batch=is_image_batch,
                visual_rope_cache=visual_rope_cache,
                sequece_packing_scales=sequece_packing_scales,
                super_scale_lengths=super_scale_lengths,
                other_info_by_scale=other_info_by_scale,
                scale_or_time_ids=scale_or_time_ids,
            ) # loss & acc_bit: [seq_len]

            # [loss reweight]
            # import pdb; pdb.set_trace()
            example_global_scales = 10
            acc_pt2scale_acc = {}
            acc_pt2scale_acc_counter = {}
            for full_pt in self.dynamic_resolution_h_w[self.h_div_w_templates[0]]['0.06M']['pt2scale_schedule']:
                full_pt = int(np.round((full_pt-1) / 4)) * 4 + 1
                if full_pt not in acc_pt2scale_acc:
                    acc_pt2scale_acc[full_pt] = [[] for _ in range(example_global_scales)]
                    acc_pt2scale_acc_counter[full_pt] = [0 for _ in range(example_global_scales)]
            
            flatten_L_list, flatten_acc_bit_list, flatten_weight_list = [], [], []
            ptr = 0
            global_scale_ind = 0
            for sample_ind, item in enumerate(sequece_packing_scales):
                full_pt = full_pts_this_batch[sample_ind]
                full_pt = int(np.round((full_pt-1) / 4)) * 4 + 1
                for si, (pt, ph, pw) in enumerate(item):
                    mul_pt_ph_pw = pt * ph * pw
                    start, end = ptr, ptr+mul_pt_ph_pw
                    ptr = end
                    loss_this_scale = loss[start:end].mean()
                    acc_this_scale = acc_bit[start:end].mean()
                    wandb_plot_index = other_info_by_scale[global_scale_ind]['wandb_plot_index']
                    acc_pt2scale_acc[full_pt][wandb_plot_index].append(acc_this_scale)
                    acc_pt2scale_acc_counter[full_pt][wandb_plot_index] += 1
                    flatten_weight_list.append(mul_pt_ph_pw ** args.reweight_loss_by_scale)
                    flatten_L_list.append(loss_this_scale)
                    flatten_acc_bit_list.append(acc_this_scale)
                    global_scale_ind += 1
            flatten_weight_list = torch.tensor(flatten_weight_list, dtype=loss.dtype, device=loss.device)
            flatten_weight_list = flatten_weight_list / flatten_weight_list.sum()
            final_loss = (torch.stack(flatten_L_list) * flatten_weight_list).sum()
            final_acc_bit = (torch.stack(flatten_acc_bit_list) * flatten_weight_list).sum()
        
        # [backward]
        final_loss.backward(retain_graph=False, create_graph=False)
        if self.zero:
            grad_norm_t = self.gpt.clip_grad_norm_(args.tclip)
        else:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(self.gpt.parameters(), args.tclip) # non zero mode
        self.gpt_opt.step()

        # update ema 
        if args.use_fsdp_model_ema and (args.model_ema_decay < 1):
            update_ema(self.gpt_ema, self.gpt, args.model_ema_decay)

        # [zero_grad]
        if stepping:
            self.gpt_opt.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:            
            acc_pt2scale_acc = sum_dict(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc = dict2list(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc_counter = dict2list(acc_pt2scale_acc_counter)

            train_loss = final_loss.item()
            train_acc = final_acc_bit.item()
            metrics = torch.tensor(flatten_acc_pt2scale_acc + flatten_acc_pt2scale_acc_counter + [grad_norm_t.item(), train_loss, train_acc, is_image_batch, valid_sequence_ratio], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            flatten_acc_pt2scale_acc, flatten_acc_pt2scale_acc_counter = metrics[:len(flatten_acc_pt2scale_acc)], metrics[len(flatten_acc_pt2scale_acc):2*len(flatten_acc_pt2scale_acc)]
            flatten_acc_pt2scale_acc = flatten_acc_pt2scale_acc / (flatten_acc_pt2scale_acc_counter + 1e-16)
            acc_pt2scale_acc = list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc)
            acc_pt2scale_acc_counter = list2dict(acc_pt2scale_acc_counter, flatten_acc_pt2scale_acc_counter)
            grad_norm_t, train_loss, train_acc, is_image_batch, valid_sequence_ratio = metrics[2*len(flatten_acc_pt2scale_acc):] / (dist.get_world_size() + 1e-16)
            if args.num_of_label_value == 1:
                key, base = 'Loss', 1
            else:
                key, base = 'Acc', 100
            metric_lg.update(L=train_loss, Acc=train_acc*base, L_i=0., Acc_i=0., L_v=0., Acc_v=0., tnm=grad_norm_t, seq_usage=valid_sequence_ratio*100.)    # todo: Accm, Acct
            wandb_log_dict = {
                'Overall/train_loss': train_loss,
                'Overall/train_acc': train_acc*base,
                'Overall/grad_norm_t': grad_norm_t,
                'Overall/logits_abs_mean': logits_norm.item(),
                'Overall/video_batch_ratio': (1-is_image_batch)*100., 
                'Overall/valid_sequence_ratio': valid_sequence_ratio*100.,
            }
            for full_pt in acc_pt2scale_acc:
                for si in range(len(acc_pt2scale_acc[full_pt])):
                    if acc_pt2scale_acc_counter[full_pt][si] > 0:
                        duration = (full_pt-1) / 4
                        prefix = f't{duration:04.1f}s/signal_{si*0.1:.01f}_{(si+1)*0.1:.01f}'
                        wandb_log_dict[f'Details/{key}/{prefix}'] = acc_pt2scale_acc[full_pt][si].item() * base
                        wandb_log_dict[f'Details/Num/{prefix}'] = acc_pt2scale_acc_counter[full_pt][si]
            wandb_utils.log(wandb_log_dict, step=g_it)
        
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(Trainer, self).__repr__().replace(Trainer.__name__, "")}'
        )
      
    def get_config(self):
        return {}
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config()}
        
        if self.zero:
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = None
        else:
            if self.using_ema:
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
        else:
            raise NotImplementedError
