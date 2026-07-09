import os
import json
import math
import bisect

import numpy as np
import torch
import torch.nn.functional as F

from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input

def get_scale_pack_info(**kwargs):
    return

def flatten_two_level_list(two_level_list):
    flatten_list = []
    for item in two_level_list:
        flatten_list.extend(item)
    return flatten_list

def shift_pt(pt, alpha):
    """shift pt (signal ratio) to lower one, recommand alpha=sqrt(height*width/256/256)"""
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1+(alpha-1)*noise_pt) # shift noise_pt to higer one
    pt = 1 - noise_pt
    return pt

def video_encode(
    vae,
    inp_B3HW,
    vae_features=None,
    device='cuda',
    args=None,
    infer_mode=False,
    rope2d_freqs_grid=None,
    dynamic_resolution_h_w=None,
    tokens_remain=9999999,
    text_lens=[],
    caption_nums=[],
    rank_vary_generator=None,
    vis_verbose=False,
    meta_list=None,
    **kwargs,
):
    if rank_vary_generator is not None:
        numpy_generator = rank_vary_generator['numpy_generator']
        torch_cuda_generator = rank_vary_generator['torch_cuda_generator']
    else:
        numpy_generator = np.random.default_rng()
        torch_cuda_generator = torch.Generator(device='cuda')
    
    if vae_features is None:
        raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW, scale_schedule=None, slice=True)
        raw_features_list = [raw_features]
        x_recon_raw = vae.decode(raw_features[-1], slice=True)
        x_recon_raw = torch.clamp(x_recon_raw, min=-1, max=1)
        print(f'raw_features[-1].shape: {raw_features[-1].shape}')
    else:
        raw_features_list = vae_features
    # raw_features_list: list of [1,d,t,h,w]:
    # import pdb; pdb.set_trace()
    gt_all_bit_indices = []
    pred_all_bit_indices = []
    var_input_list = []
    sequece_packing_scales = [] # with trunk
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))
    visual_rope_cache_list = []
    other_info_by_scale = []
    scale_lengths = []
    with torch.amp.autocast('cuda', enabled = False):
        for example_ind, raw_features in enumerate(raw_features_list):
            meta = meta_list[example_ind]
            gt_all_bit_indices.append([])
            pred_all_bit_indices.append([])
            var_input_list.append([])
            visual_rope_cache_list.append([])
            other_info_by_scale.append([])
            B, C, T, H, W = raw_features[-1].shape
            h_div_w = H / W
            mapped_h_div_w_template = h_div_w_template_list[np.argmin(np.abs(h_div_w-h_div_w_template_list))]
            pn = meta['pn']
            if meta['first_frame_condition']:
                scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][pn]['pt2scale_schedule'][T-1]
            else:
                scale_schedule = dynamic_resolution_h_w[mapped_h_div_w_template][pn]['pt2scale_schedule'][T]
            if not infer_mode:
                next_tokens_remain = tokens_remain - T * H * W - args.add_scale_token - text_lens[example_ind]
                if next_tokens_remain < 0:
                    break
                tokens_remain = next_tokens_remain
                scale_lengths.append(T * H * W + text_lens[example_ind] + args.add_scale_token)
            preserve_scale_schedule = []
            preserve_scale_schedule.append(scale_schedule[0])
            target = raw_features[0]
            if not infer_mode and args.log_norm_sigma > 0:
                spt = torch.sigmoid(torch.randn(1, generator=torch_cuda_generator, device=target.device) * args.log_norm_sigma + args.log_norm_mean).item()
                spt = shift_pt(spt, args.alpha)
            else:
                spt = shift_pt(numpy_generator.random(), args.alpha)
            
            if args.refine_mode in ['ar_discrete_GRN_ind']:
                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2index_label
                labels = raw_feature2index_label(target, hbq_round=args.hbq_round) # [B, hbq_round * d, t, h, w]
                classes = 2**args.hbq_round
            elif args.refine_mode in ['ar_discrete_GRN_bit']:
                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
                labels = raw_feature2bit_label(target, hbq_round=args.hbq_round) # [B, hbq_round * d, t, h, w]
                classes = 2
                
            random_labels = torch.randint(0, classes, size=labels.shape, generator=torch_cuda_generator, device=labels.device, dtype=labels.dtype) # random 0 or 1 labels
            random_mask = torch.rand(size=labels.shape, generator=torch_cuda_generator, device=labels.device, dtype=target.dtype) < spt
            mixed_xt = torch.where(random_mask, labels, random_labels) # [B, hbq_round * d, t, h, w]
            precise_spt = random_mask.float().mean()
            wandb_plot_index = min(9, int(precise_spt / 0.1)) # 0~9
            
            # get visual rope
            if not infer_mode:
                if meta['first_frame_condition']:
                    visual_rope_cache_list[-1].append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule[0], device, mapped_h_div_w_template, t_offset=1)) # (2, 1, 1, 1, pt*ph*pw, dim_div_2)
                    visual_rope_cache_list[-1].append(get_visual_rope_embeds(rope2d_freqs_grid, (1, H, W), device, mapped_h_div_w_template, t_offset=0))
                    visual_rope_cache_list[-1] = [torch.cat(visual_rope_cache_list[-1], dim=-2)]
                else:
                    visual_rope_cache_list[-1].append(get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule[0], device, mapped_h_div_w_template, t_offset=0)) # (2, 1, 1, 1, pt*ph*pw, dim_div_2)

            # get visual tokens
            visual_token_dim = mixed_xt.shape[1] * classes
            
            if not infer_mode and meta['first_frame_condition']:
                first_frame_labels = labels[:,:,:1] # [B, hbq_round * d, 1, h, w]
                first_frame_tokens = multiclass_labels2onehot_input(first_frame_labels, classes).reshape(1, visual_token_dim, -1).permute(0, 2, 1)
                cur_visual_tokens = multiclass_labels2onehot_input(mixed_xt[:,:,1:], classes).reshape(1, visual_token_dim, -1).permute(0, 2, 1)
                cur_visual_tokens = torch.cat((cur_visual_tokens, first_frame_tokens), dim=1)
                indices = labels[:,:,1:]
            else:
                cur_visual_tokens = multiclass_labels2onehot_input(mixed_xt, classes).reshape(1, visual_token_dim, -1).permute(0, 2, 1)
                indices = labels
            indices = indices.type(torch.long).permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
            gt_all_bit_indices[-1].append(indices)
            var_input_list[-1].append(cur_visual_tokens)
            other_info_by_scale[-1].append(
                {
                    'largest_scale': scale_schedule[-1], 
                    'wandb_plot_index': wandb_plot_index, 
                    'cur_bits': indices.shape[-1],
                    'cur_lvl': args.detail_num_lvl,
                    'scale_token_id': precise_spt,
                    'predict_tokens': np.prod(scale_schedule[0]),
                    'all_tokens': scale_lengths[-1] if len(scale_lengths) else -1,
                    'first_frame_condition': meta['first_frame_condition'],
                }
            )
            sequece_packing_scales.append(preserve_scale_schedule)

    gt_all_bit_indices = flatten_two_level_list(gt_all_bit_indices)
    pred_all_bit_indices = flatten_two_level_list(pred_all_bit_indices)
    var_input_list = flatten_two_level_list(var_input_list)
    visual_rope_cache_list = flatten_two_level_list(visual_rope_cache_list)
    other_info_by_scale = flatten_two_level_list(other_info_by_scale)

    if infer_mode:
        return [labels, target], x_recon_raw, [target], None, None, None
        
    gt_ms_idx_Bl = []
    for item in gt_all_bit_indices:
        _, tt, hh, ww, dd = item.shape
        item = item.reshape(B, tt*hh*ww, dd)
        gt_ms_idx_Bl.append(item)
    gt_BLC = gt_ms_idx_Bl # torch.cat(gt_ms_idx_Bl, 1).contiguous().type(torch.long)
    x_BLC = var_input_list
    x_BLC_mask = None
    scale_or_time_ids = None
    return x_BLC, x_BLC_mask, scale_or_time_ids, gt_BLC, pred_all_bit_indices, visual_rope_cache_list, sequece_packing_scales, scale_lengths, other_info_by_scale

def video_decode(
    vae,
    all_indices,
    scale_schedule,
    label_type,
    args=None,
    noise_list=None,
    trunc_scales=-1,
    **kwargs,
):  
    if trunc_scales < 0:
        summed_codes = all_indices[-1]
    else:
        summed_codes = all_indices[trunc_scales-1]
    x_recon = vae.decode(summed_codes, slice=True)
    x_recon = torch.clamp(x_recon, min=-1, max=1)
    x_recon_256 = None
    return x_recon, x_recon_256

def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, device=None, mapped_h_div_w_template=None, t_offset=0):
    # freqs_frames: (2, max_frames, dim_div_2 / 3)
    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    max_height = rope2d_freqs_grid['freqs_height'].shape[1]
    max_width = rope2d_freqs_grid['freqs_width'].shape[1]
    extreme_h_div_w = 3
    assert mapped_h_div_w_template <= extreme_h_div_w
    extreme_h = max_height
    extreme_w = extreme_h / extreme_h_div_w
    upw = np.sqrt(extreme_h * extreme_w / mapped_h_div_w_template)
    uph = mapped_h_div_w_template * upw
    uph, upw = int(uph), int(upw)
    pt, ph, pw = scale_schedule
    assert ph <= uph and pw <= upw
    f_frames = rope2d_freqs_grid['freqs_frames'][:, t_offset:t_offset+pt]
    f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
    f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
    rope_embeds = torch.cat([
        f_frames[   :,     :,  None,   None,   :].expand(-1, -1, ph, pw, -1),
        f_height[   :,  None,      :,  None,   :].expand(-1,  pt,-1, pw, -1),
        f_width[   :,  None,   None,      :,   :].expand(-1,  pt,ph, -1, -1),
    ], dim=-1)  # (2, pt, ph, pw, dim_div_2)
    rope_embeds = rope_embeds.reshape(2, 1, 1, 1, pt*ph*pw, -1)  # (2, 1, 1, 1, pt*ph*pw, dim_div_2)
    return rope_embeds
