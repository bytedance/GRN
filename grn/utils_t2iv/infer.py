import hashlib
import os
import os.path as osp
import re
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision.transforms.functional import to_tensor

torch._dynamo.config.cache_size_limit = 64

from grn.models.basic import *
from grn.models.grn import GRN
from grn.models.umt5.t5 import T5EncoderModel

def extract_key_val(text):
    return {k: v.lstrip() for k, v in re.findall(r'<(.+?):(.+?)>', text)}

def encode_prompt(t5_path, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, args=None):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f't5 encode prompt: {prompt}')
    text_encoder.model.to(args.other_device)
    text_features = text_encoder([prompt], args.other_device)
    lens = [len(item) for item in text_features]
    cu_seqlens_k = [0]
    for len_i in lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
    Ltext = max(lens)
    kv_compact = torch.cat(text_features, dim=0).float()
    kv_compact = kv_compact.to(args.other_device)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def aug_with_positive_prompt(prompt):
    keys = {'man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
            'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter'}
    if any(key in prompt for key in keys):
        prompt += '. very smooth faces, good looking faces, face to the camera, perfect facial features'
    return prompt

def gen_one_example(
    model, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt="",
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_latent_dim=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    input_use_interplote_up=False,
    args=None,
    get_visual_rope_embeds=None,
    context_info=None,
    noise_list=None,
    return_summed_code_only=False,
    class_token_id=0,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    
    text_cond_tuple = []
    for prompt_str in prompt:
        text_cond_tuple.append(encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt_str, enable_positive_prompt, args=args))
    negative_label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, negative_prompt, enable_positive_prompt=False, args=args)
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        out = model.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_latent_dim=vae_latent_dim, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            input_use_interplote_up=input_use_interplote_up,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=noise_list,
            return_summed_code_only=return_summed_code_only,
            class_token_id=class_token_id,
        )
        _, pred_multi_scale_bit_labels, img_list = out
            
    print(f"cost: {time.time() - sstt}, model cost={time.time() - stt}")
    img = img_list[0]
    return img

def get_prompt_id(prompt):
    return hash_string(prompt)

def load_tokenizer(t5_path ='', device='cuda'):
    print(f'[Loading tokenizer and text encoder]')
    if isinstance(device, str):
        device = torch.device(device)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=osp.join(t5_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=osp.join(t5_path, 'umt5-xxl'),
        enable_fsdp=False)
    text_tokenizer = text_encoder.tokenizer
    return text_tokenizer, text_encoder

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(np.round(tgt_w / (width / height)))
    else:
        resized_height = tgt_h
        resized_width = int(np.round((width / height) * tgt_h))
    pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im * 2 - 1

def hash_string(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, _ = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = ((recons_img + 1) / 2).clamp(0, 1)
    recons_img = recons_img.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
    gt_img = ((inp[0] + 1) / 2).clamp(0, 1)
    gt_img = gt_img.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices


def load_transformer(vae, args):
    device = torch.device(args.other_device)
    model_path = args.model_path
    if not model_path:
        state_dict = None
    elif args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
        state_dict = torch.load(slim_model_path, map_location=device)

    print(f'[Loading Model]')
    # Check if device is CUDA before enabling autocast
    if device.type == 'cuda':
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            model = create_model(
                args.model,
                vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=False,
                num_of_label_value=args.num_of_label_value,
                rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
                pn=args.pn,
                apply_spatial_patchify=args.apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=args.train_h_div_w_list,
                dynamic_scale_schedule=args.dynamic_scale_schedule,
                video_frames=args.video_frames,
                other_args=args,
            ).to(device=device)
    else:
        with torch.no_grad():
            model = create_model(
                args.model,
                vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=False,
                num_of_label_value=args.num_of_label_value,
                rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
                pn=args.pn,
                apply_spatial_patchify=args.apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=args.train_h_div_w_list,
                dynamic_scale_schedule=args.dynamic_scale_schedule,
                video_frames=args.video_frames,
                other_args=args,
            ).to(device=device)
    print(f'[you selected model with {args.model}] model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B, bf16={args.bf16}')
    if args.bf16:
        for block in model.unregistered_blocks:
            block.bfloat16()
    model.eval()
    model.requires_grad_(False)
    # Only call cuda() if device is CUDA
    if device.type == 'cuda':
        model.cuda()
        torch.cuda.empty_cache()
    print(f'[Load model weights]')
    if state_dict:
        print(model.load_state_dict(state_dict, strict=True))
    return model

def images2video(ndarray_image_list, fps=24, save_filepath='tmp.mp4'):
    # ndarray_image_list: bgr sequence
    save_dir = osp.dirname(save_filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if len(ndarray_image_list) == 1:
        save_filepath = osp.splitext(save_filepath)[0] + '.jpg'
        cv2.imwrite(save_filepath, ndarray_image_list[0]) # bgr
        print(f"Image saved as {osp.abspath(save_filepath)}")
    else:
        # imageio takes rgb, so convert bgr to rgb
        imageio.mimsave(save_filepath, ndarray_image_list[..., ::-1], fps=fps) 
        print(f"Video saved as {osp.abspath(save_filepath)}")

def imgs_tensor2uint8_imgs(imgs_tensor):
    imgs_tensor = imgs_tensor.permute(1, 2, 3, 0) # [c,t,h,w] -> [t,h,w,c]
    imgs_tensor = ((imgs_tensor + 1) / 2).clamp(0, 1)
    return imgs_tensor.mul(255).to(torch.uint8).flip(dims=(3,)).cpu().numpy()