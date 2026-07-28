import glob
import os
import pickle
import random
import re
import time
from functools import partial
from os import path as osp
from typing import List, Tuple, Union
import json
import itertools
import hashlib
import copy
import collections
import math

import tqdm
import numpy as np
import torch
import pandas as pd
from decord import VideoReader
from PIL import Image as PImage
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor, hflip
from torchvision.transforms import transforms, InterpolationMode
from torch.utils.data import Dataset, DataLoader
import torch.distributed as tdist
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils.video_decoder import EncodedVideoDecord
from grn.utils.compress_tokens import load_packed_tensor
from transformers import AutoTokenizer

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def normalize(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def prepend_motion_score(prompt, motion_score):
    return f'<<<motion_score: {round(motion_score):.1f}>>> {prompt}'

class VideoReaderWrapper(VideoReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seek(0)
    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames

class JointViDataset(Dataset):
    def __init__(
        self,
        meta_folders: str = '',
        meta_folder_repeats: str = '',
        buffersize: int = 1000000 * 300,
        seed: int = 0,
        pn: str = '',
        video_fps: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        dataloader_workers: int = 2,
        enable_dynamic_length_prompt: bool = True,
        shuffle: bool = True,
        short_prob: float = 0.2,
        verbose=False,
        temp_dir= "/dev/shm",
        hdfs_mode='read',
        other_args=None,
        **kwargs,
    ):
        self.meta_folders = json.loads(meta_folders)
        self.meta_folder_repeats = json.loads(meta_folder_repeats)
        self.meta_folder_identifiers = json.loads(other_args.meta_folder_identifiers)
        self.pn_list = json.loads(other_args.pn_list)
        self.pn_probs = json.loads(other_args.pn_probs)
        assert len(self.meta_folders) == len(self.meta_folder_repeats), f'{len(self.meta_folders)} != {len(self.meta_folder_repeats)}'
        assert len(self.meta_folders) == len(self.meta_folder_identifiers), f'{len(self.meta_folders)} != {len(self.meta_folder_identifiers)}'
        self.verbose = verbose
        self.buffer_size = buffersize
        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = 0
        self.global_worker_id = 0
        self.short_prob = short_prob
        self.dataloader_workers = max(1, dataloader_workers)
        self.shuffle = shuffle
        self.global_workers = self.num_replicas * self.dataloader_workers
        self.seed = seed
        self.text_tokenizer = other_args.text_tokenizer
        self.feature_extraction = other_args.only_images4extract_feats # < 0 # no sequence packing, for feature extraction
        self.epoch_generator = None
        self.epoch_rank_generator = None
        self.other_args = other_args
        self.pair_input = other_args.pair_input
        self.drop_long_video = other_args.drop_long_video
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        self.set_epoch_generator(other_args.epoch)
        self.temporal_compress_rate = other_args.temporal_compress_rate
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames) # here video_frames is the max video frames
        self.video_fps = video_fps
        self.min_training_duration = (other_args.min_video_frames-1) // self.video_fps
        self.max_training_duration = (other_args.video_frames-1) // self.video_fps
        self.c2i = self.other_args.add_class_token > 0
        if self.c2i:
            self.c2i_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(288, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
                transforms.RandomCrop((256, 256)),
                transforms.ToTensor(), 
                normalize,
            ])
        else:
            self.c2i_transform = None
        self.print(f"{self.rank=} dataset {self.seed=}, {self.h_div_w_templates=} {self.min_training_duration=} {self.max_training_duration=}, cache_check_mode={self.other_args.cache_check_mode}")
        self.token_cache_dir = other_args.token_cache_dir
        self.use_vae_token_cache = other_args.use_vae_token_cache
        self.allow_online_vae_feature_extraction = other_args.allow_online_vae_feature_extraction
        self.use_text_token_cache = other_args.use_text_token_cache
        self.max_video_frames = other_args.video_frames
        self.cached_video_frames = other_args.cached_video_frames # cached max video frames
        self.down_size_limit = other_args.down_size_limit
        self.video_caption_type = other_args.video_caption_type
        self.train_max_token_len = other_args.train_max_token_len
        self.duration_resolution = other_args.duration_resolution
        self.device = other_args.device
        print(f'self.down_size_limit: {self.down_size_limit}')
        self.hdfs_mode = hdfs_mode
        self.max_text_len = other_args.tlen
        self.temp_dir = temp_dir.rstrip("/")
        self.mapped_duration2metas, self.mapped_duration2freqs = self.get_mapped_duration2metas()
        self.batches = self.form_batches(self.mapped_duration2metas)
        print(f'{num_replicas=}, {rank=}, {dataloader_workers=}, {len(self.batches)=}, {self.drop_long_video=} {self.max_text_len=} self.batches[:10]={self.batches[:10]}')

    def print(self, string):
        if self.feature_extraction:
            print(string)
        else:
            print(string, force=True)
    
    def get_captions_lens(self, captions):
        if self.other_args.text_tokenizer_type == 'flan_t5':
            tokens = self.other_args.text_tokenizer(text=captions, max_length=self.other_args.text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')
            mask = tokens.attention_mask.cuda(non_blocking=True)
            lens: List[int] = mask.sum(dim=-1).tolist()
        else: # umt5-xxl
            ids, mask = self.other_args.text_tokenizer( captions, return_mask=True, add_special_tokens=True)
            lens = mask.gt(0).sum(dim=1).tolist()
        return lens
        
    def get_video_caption(self, meta, mapped_duration):
        if 'tarsier2_caption' not in meta:
            caption = self.epoch_rank_generator.choice(meta['caption'])['content']
        else:
            caption_type = 'tarsier2_caption'
            if ('MiniCPM_V_2_6_caption' in meta) and meta['MiniCPM_V_2_6_caption']:
                caption_type = self.epoch_rank_generator.choice(['tarsier2_caption', 'MiniCPM_V_2_6_caption'])
            caption = meta[caption_type]
        if self.enable_dynamic_length_prompt and (self.epoch_rank_generator.random() < self.other_args.short_cap_prob):
            caption = self.random_drop_sentences(caption, min_sentences=2)
        if 'quality_prompt' in meta:
            caption = caption + ' ' + meta['quality_prompt']
        if meta['first_frame_condition']:
            caption = '<I2V>' + caption
        else:
            caption = '<T2V>' + caption
        assert caption
        return caption
    
    def get_image_caption(self, meta):
        caption = meta['long_caption']
        if not meta['long_caption']:
            caption = meta['text']
        else:
            if self.epoch_rank_generator.random() < self.other_args.short_cap_prob:
                if meta['text']:
                    caption = meta['text']
                elif ('InternVL' in meta['long_caption_type']):
                    caption = self.random_drop_sentences(meta['long_caption'], min_sentences=2)
        caption = '<T2I>' + caption
        assert caption
        return caption

    def sample_pn(self, meta, pn_list, pn_probs):
        if ('height' in meta) and ('width' in meta):
            real_pn = meta['height'] * meta['width'] / 1000000
            valid_pn_list, valid_pn_probs = [], []
            for pn, pn_prob in zip(pn_list, pn_probs):
                if real_pn > 0.7 * float(pn[:-1]):
                    valid_pn_list.append(pn)
                    valid_pn_probs.append(pn_prob)
            if not len(valid_pn_list):
                valid_pn_list = pn_list[:1]
                valid_pn_probs = pn_probs[:1]
        else:
            valid_pn_list = pn_list
            valid_pn_probs = pn_probs
        valid_pn_probs = np.array(valid_pn_probs)
        valid_pn_probs = valid_pn_probs / valid_pn_probs.sum()
        return self.epoch_rank_generator.choice(valid_pn_list, p=valid_pn_probs)

    def get_mapped_duration2metas(self):
        part_filepaths = []
        part_file2identifier = {}
        for meta_folder, meta_folder_repeat, meta_folder_identifier in zip(self.meta_folders, self.meta_folder_repeats, self.meta_folder_identifiers):
            tmp_part_filepaths = sorted(glob.glob(osp.join(meta_folder, '*/*.jsonl')))
            self.epoch_generator.shuffle(tmp_part_filepaths)
            if meta_folder_repeat > 1:
                tmp_part_filepaths = tmp_part_filepaths * int(np.ceil(meta_folder_repeat))
            tmp_part_filepaths = tmp_part_filepaths[:int(len(tmp_part_filepaths)*meta_folder_repeat)]
            for file in tmp_part_filepaths: part_file2identifier[file] = meta_folder_identifier
            part_filepaths.extend(tmp_part_filepaths)
        self.epoch_generator.shuffle(part_filepaths)
        self.print(f'{self.rank=} jsonls sample: {part_filepaths[:4]}')
        if self.num_replicas > 1:
            part_filepaths = part_filepaths[self.rank::self.num_replicas]
        
        mapped_duration2metas = {}
        pbar = tqdm.tqdm(total=len(part_filepaths))
        total, corrupt = 0, 0
        stop_read = False
        rough_h_div_w = self.h_div_w_templates[np.argmin(np.abs((9/16-self.h_div_w_templates)))]
        meta_folder2freqs = collections.defaultdict(int)
        for part_filepath in part_filepaths:
            file_quality_prompt = part_file2identifier[part_filepath]
            meta_folder = osp.dirname(osp.dirname(part_filepath))
            if stop_read:
                break
            pbar.update(1)
            try:
                with open(part_filepath, encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f'{part_filepath=} Error: {e}')
                lines = []
            for line in lines:
                total += 1
                try:
                    meta = json.loads(line)
                except Exception as e:
                    print(e)
                    corrupt += 1
                    print(e, corrupt, total, corrupt/total)
                    continue
                if file_quality_prompt: # override quality prompt
                    meta['quality_prompt'] = file_quality_prompt
                if ('height' in meta) and ('width' in meta):
                    cur_h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((meta['height']/meta['width']-self.h_div_w_templates)))]
                else:
                    cur_h_div_w_template = rough_h_div_w
                if 'h_div_w' in meta:
                    del meta['h_div_w']
                meta['first_frame_condition'] = False
                meta['pn'] = self.sample_pn(meta, self.pn_list, self.pn_probs)
                if 'video_path' in meta:
                    if self.epoch_rank_generator.random() < self.other_args.i2v_ratio:
                        meta['first_frame_condition'] = True
                    begin_frame_id, end_frame_id, fps = meta['begin_frame_id'], meta['end_frame_id'], meta['fps']
                    real_duration = (end_frame_id - begin_frame_id) / fps
                    mapped_duration = int(np.round(real_duration / self.duration_resolution)) * self.duration_resolution
                    if mapped_duration < self.min_training_duration:
                        continue
                    if mapped_duration > self.max_training_duration:
                        if self.drop_long_video:
                            continue
                        else:
                            mapped_duration = self.max_training_duration
                    if self.other_args.use_clipwise_caption:
                        meta['caption'] = [
                            meta['caption-InternVL2.0'],
                            self.get_video_caption(meta, mapped_duration),
                        ]
                    else:
                        meta['caption'] = [self.get_video_caption(meta, mapped_duration)]
                    sample_frames = int(mapped_duration * self.video_fps + 1)
                    pt = (sample_frames-1) // self.temporal_compress_rate + 1
                    scale_schedule = self.dynamic_resolution_h_w[cur_h_div_w_template][meta['pn']]['pt2scale_schedule'][pt]
                    meta['sample_frames'] = sample_frames
                elif 'image_path' in meta:
                    mapped_duration = -1
                    scale_schedule = self.dynamic_resolution_h_w[cur_h_div_w_template][meta['pn']]['pt2scale_schedule'][1]
                    meta['caption'] = [self.get_image_caption(meta)]
                # random set caption to "" for classifier-free guidance
                # refer to: https://github.com/PixArt-alpha/PixArt-alpha/blob/master/train_scripts/train_diffusers.py#L67
                for caption_ind in range(len(meta['caption'])):
                    if self.epoch_rank_generator.random() < self.other_args.drop_condition_prob:
                        meta['caption'][caption_ind] = ""
                if mapped_duration not in mapped_duration2metas:
                    mapped_duration2metas[mapped_duration] = []
                
                # get cum_text_visual_tokens
                cum_visual_tokens = []
                preserve_scale_inds = {}
                assert len(scale_schedule) == len(self.other_args.video_scale_probs), f'{len(scale_schedule)=} {len(self.other_args.video_scale_probs)=}'
                for scale_ind, scale in enumerate(scale_schedule):
                    if self.epoch_rank_generator.random() < self.other_args.video_scale_probs[scale_ind]:
                        preserve_scale_inds[scale_ind] = True
                        tokens_this_scale = np.array(scale).prod(-1) + self.other_args.add_scale_token
                        cum_visual_tokens.append(tokens_this_scale)
                cum_visual_tokens = np.array(cum_visual_tokens).cumsum()
                meta['cum_text_visual_tokens'] = cum_visual_tokens
                meta['preserve_scale_inds'] = preserve_scale_inds
                
                if self.other_args.cache_check_mode == 1: # check at the begining
                    if self.exists_cache_file(meta):
                        mapped_duration2metas[mapped_duration].append(meta)
                        meta_folder2freqs[meta_folder] += 1
                elif self.other_args.cache_check_mode == -1: # select unexist, used for token cache
                    if not self.exists_cache_file(meta):
                        mapped_duration2metas[mapped_duration].append(meta)
                        meta_folder2freqs[meta_folder] += 1
                else:
                    mapped_duration2metas[mapped_duration].append(meta)
                    meta_folder2freqs[meta_folder] += 1
                
                total_metas = sum([len(item) for item in mapped_duration2metas.values()])
                if (self.other_args.restrict_data_size > 0) and (total_metas > self.other_args.restrict_data_size / self.num_replicas):
                    stop_read = True
                    break
                
        # print meta folder2freqs
        for meta_folder, freq in sorted(meta_folder2freqs.items(), key=lambda x: x[1], reverse=True):
            print(f'{meta_folder=}, {freq=}, {freq / total_metas * 100:.1f}%')
        
        # set mapped_duration2freqs
        mapped_duration2freqs = {}
        for mapped_duration in sorted(mapped_duration2metas.keys()):
            mapped_duration2freqs[mapped_duration] = len(mapped_duration2metas[mapped_duration])

        for mapped_duration in mapped_duration2metas.keys():
            freqs = mapped_duration2freqs[mapped_duration]
            assert len(mapped_duration2metas[mapped_duration]) >= freqs
            self.epoch_rank_generator.shuffle(mapped_duration2metas[mapped_duration])
            mapped_duration2metas[mapped_duration] = mapped_duration2metas[mapped_duration][:freqs]
            # append text tokens
            skip_count_text_token = self.other_args.skip_count_text_token or self.other_args.add_class_token > 0
            mapped_duration2metas[mapped_duration] = self.append_text_tokens(mapped_duration2metas[mapped_duration], skip_count_text_token=skip_count_text_token)

        total_metas = sum([len(item) for item in mapped_duration2metas.values()])
        for mapped_duration in sorted(mapped_duration2freqs.keys()):
            freq = mapped_duration2freqs[mapped_duration]
            proportion = freq / total_metas * 100
            print(f'{mapped_duration=}, {freq=}, {proportion=:.1f}%')
        return mapped_duration2metas, mapped_duration2freqs

    def append_text_tokens(self, metas, skip_count_text_token=False, bucket_size=100):
        t1 = time.time()
        pbar = tqdm.tqdm(total=len(metas) // bucket_size + 1, desc='append text tokens')
        valid_metas = []
        for bucket_id in range(len(metas) // bucket_size + 1):
            pbar.update(1)
            start = bucket_id * bucket_size
            end = min(start + bucket_size, len(metas))
            if start >= end:
                break
            captions = []
            caps_per_meta = []
            for i in range(start, end):
                captions.extend(metas[i]['caption'])
                caps_per_meta.append(len(metas[i]['caption']))
            assert len(captions), f'{len(captions)=}'
            if skip_count_text_token:
                lens = [0 for _ in range(len(captions))]
            else:
                lens = self.get_captions_lens(captions)
                lens = np.clip(np.array(lens), a_min=0, a_max=self.max_text_len)
            ptr = 0
            for i in range(start, end):
                text_tokens = sum(lens[ptr:ptr+caps_per_meta[i-start]])
                ptr += caps_per_meta[i-start]
                metas[i]['text_tokens'] = text_tokens
                metas[i]['cum_text_visual_tokens'] = metas[i]['cum_text_visual_tokens'] + metas[i]['text_tokens']
                metas[i]['text_visual_tokens'] = metas[i]['cum_text_visual_tokens'][-1]
                if metas[i]['text_visual_tokens'] <= self.train_max_token_len * self.other_args.dense_ratio4seqpack:
                    valid_metas.append(metas[i])
        t2 = time.time()
        print(f'append text tokens: {t2-t1:.1f}s')
        return valid_metas

    def exists_cache_file(self, meta):
        pn = meta['pn']
        if 'image_path' in meta:
            return osp.exists(self.get_image_cache_file(meta['image_path'], pn))
        else:
            if '/vdataset/clip' in meta['video_path']: # clip
                cache_file = self.get_video_cache_file(meta['video_path'], 0, meta['end_frame_id']-meta['begin_frame_id'], self.video_fps, pn)
            else:
                cache_file = self.get_video_cache_file(meta['video_path'], meta['begin_frame_id'], meta['end_frame_id'], self.video_fps, pn)
            return osp.exists(cache_file)
    
    def form_batches(self, mapped_duration2metas):
        examples = []
        for mapped_duration in sorted(mapped_duration2metas.keys()):
            for example_ind in range(len(mapped_duration2metas[mapped_duration])):
                text_visual_tokens = mapped_duration2metas[mapped_duration][example_ind]['text_visual_tokens']
                examples.append((mapped_duration, example_ind, text_visual_tokens))
        examples = sorted(examples, key=lambda x: -x[2])
        max_text_visual_tokens = examples[0][2] if len(examples) else 0
        assert self.train_max_token_len >= max_text_visual_tokens, f'{self.train_max_token_len=} should >= {max_text_visual_tokens=}'
        self.print(f'{self.rank=} {self.mapped_duration2freqs=} form_batches details: {self.rank=} examples={examples[:20]}')
        
        st = time.time()
        if self.feature_extraction or self.pair_input: # no sequence packing, for feature extraction or dpo training
            batches = [[item[:2]] for item in examples]
        else:
            batches = []
            left_ptr, right_ptr = 0, len(examples)-1
            while left_ptr <= right_ptr:
                tokens_remain = self.train_max_token_len
                tmp_batch = []
                while left_ptr <= right_ptr and (tokens_remain - examples[left_ptr][2] >= 0):
                    tokens_remain = tokens_remain - examples[left_ptr][2]
                    tmp_batch.append((left_ptr, examples[left_ptr][2]))
                    left_ptr += 1
                while left_ptr <= right_ptr and (tokens_remain - examples[right_ptr][2] >= 0):
                    tokens_remain = tokens_remain - examples[right_ptr][2]
                    tmp_batch.append((right_ptr, examples[right_ptr][2]))
                    right_ptr -= 1
                if len(tmp_batch):
                    tmp_batch = sorted(tmp_batch, key=lambda x: -x[1])
                    # total_tokens = sum(item[1] for item in tmp_batch)
                    # if total_tokens / self.train_max_token_len < 0.7:
                    #     import pdb; pdb.set_trace()
                    batches.append([examples[ptr][:2] for (ptr, _) in tmp_batch])
                    if len(batches) % 1000 == 0:
                        print(f'form {len(batches)} batches, len(metas)={len(examples)}')
        print(f'[data preprocess] form_batches done, got {len(batches)} batches, cost {time.time()-st:.2f}s')
        self.epoch_rank_generator.shuffle(batches)
        print(f'[data preprocess] shuffle batches done')
        batch_num = len(batches)
        try:
            if self.num_replicas > 1:
                batch_num = torch.tensor([batch_num], device=self.device)
                if tdist.is_initialized():
                    tdist.all_reduce(batch_num, op=tdist.ReduceOp.MIN)
                batch_num = batch_num.item()
        except Exception as e:
            print(e)
        batches = batches[:batch_num]
        print(f'[data preprocess] aligned batch number among gpus, got {batch_num} batches')
        return batches
    
    def set_epoch_generator(self, epoch):
        self.epoch = epoch
        self.epoch_generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch_rank_generator = np.random.default_rng(self.seed + self.epoch + self.rank)

    def __getitem__(self, batch_ind_ptr):
        try:
            batch_info = self.batches[batch_ind_ptr%len(self.batches)]
            batch_data = []
            for (mapped_duration, example_ind) in batch_info:
                ret = False
                repeat_times = 0
                mapped_duration_metas = self.mapped_duration2metas[mapped_duration]
                while not ret:
                    example_ind = example_ind % len(mapped_duration_metas)
                    meta = mapped_duration_metas[example_ind]
                    if 'video_path' in meta:
                        if self.pair_input:
                            ret, model_input = self.prepare_pair_video_input(meta)
                        else:
                            ret, model_input = self.prepare_video_input(meta)
                    elif 'image_path' in meta:
                        if self.pair_input:
                            ret, model_input = self.prepare_pair_image_input(meta)
                        else:
                            ret, model_input = self.prepare_image_input(meta)
                    if ret:
                        if self.pair_input:
                            batch_data.extend(model_input)
                        else:
                            batch_data.append(model_input)
                    else: # Handle corrupt example in a batch, just try to read the next one
                        example_ind = example_ind + 1
                        repeat_times += 1
                        if repeat_times % 20 == 0: # Too many corrupt files, switch to another batch
                            self.print(f'Caution! I have repeat {repeat_times} times to read a video/image, but still failed to read it. {example_ind=} {meta=}')
                            return self.__getitem__(batch_ind_ptr+1)
            
            images, raw_features_bcthw, feature_cache_files4images  = [], [], []
            text_feature_cache_files = []
            addition_pn_images = {}
            batch_data4images, batch_data4raw_features = [], []
            for item in batch_data:
                if item['raw_features_cthw'] is None:
                    images.append(item['img_T3HW'].permute(1,0,2,3)) # # tchw -> cthw
                    for key in item:
                        if key.startswith('img_T3HW_'):
                            if key not in addition_pn_images:
                                addition_pn_images[key] = []
                            addition_pn_images[key].append(item[key].permute(1,0,2,3))
                    feature_cache_files4images.append(item['feature_cache_file'])
                    batch_data4images.append(item)
                else:
                    raw_features_bcthw.append(item['raw_features_cthw'])
                    batch_data4raw_features.append(item)
            batch_data4images_raw_features = batch_data4images + batch_data4raw_features
            captions = [item['text_input'] for item in batch_data4images_raw_features]
            text_feature_cache_files = [item['text_feature_cache_file'] for item in batch_data4images_raw_features]
            meta_list = [item['meta'] for item in batch_data4images_raw_features]
            return {
                'captions': captions, 
                'images': images, 
                'addition_pn_images': addition_pn_images,
                'feature_cache_files4images': feature_cache_files4images,
                'raw_features_bcthw': raw_features_bcthw, 
                'text_cond_tuple': None,
                'text_feature_cache_files': text_feature_cache_files,
                'meta_list': meta_list,
                'media': 'videos',
            }
        except Exception as e:
            print(f'get item error: {e}')
            return self.__getitem__(batch_ind_ptr+1)


    def prepare_image_input(self, info) -> Tuple:
        try:
            img_path, text_input = osp.abspath(info['image_path']), info['caption']
            img_T3HW, raw_features_cthw, feature_cache_file, text_features_lenxdim, text_feature_cache_file = [None] * 5
            if self.use_vae_token_cache:
                feature_cache_file = self.get_image_cache_file(img_path, info['pn'])
                if osp.exists(feature_cache_file):
                    try:
                        raw_features_cthw = self.load_visual_token(feature_cache_file)
                    except Exception as e:
                        print(f'load cache file error: {e}')
                        os.remove(feature_cache_file)
                if raw_features_cthw is None and (not self.allow_online_vae_feature_extraction):
                    return False, None
            if raw_features_cthw is None:
                with open(img_path, 'rb') as f:
                    img: PImage.Image = PImage.open(f)
                    w, h = img.size
                    h_div_w = h / w
                    h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w-self.h_div_w_templates)))]
                    tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][info['pn']]['pixel']
                    img = img.convert('RGB')
                    if self.c2i:
                        img_T3HW = self.c2i_transform(img)
                    else:
                        img_T3HW = transform(img, tgt_h, tgt_w)
                    img_T3HW = img_T3HW.unsqueeze(0)
                    assert img_T3HW.shape == (1, 3, tgt_h, tgt_w)
            data_item = {
                'text_input': text_input,
                'img_T3HW': img_T3HW,
                'raw_features_cthw': raw_features_cthw,
                'feature_cache_file': feature_cache_file,
                'text_features_lenxdim': text_features_lenxdim,
                'text_feature_cache_file': text_feature_cache_file,
                'meta': info,
            }
            return True, data_item
        except Exception as e:
            print(f'prepare_image_input error: {e}')
            return False, None

    def prepare_pair_image_input(self, info) -> Tuple:
        pass
        
    def prepare_pair_video_input(self, info) -> Tuple:
        win_flag, win_data_item = self.prepare_video_input(copy.deepcopy(info))
        
        info['video_path'] = info['lose_video_path']
        lose_flag, lose_data_item = self.prepare_video_input(info)

        flag = win_flag and lose_flag
        return flag, [win_data_item, lose_data_item]

    def load_visual_token(self, feature_cache_file):
        raw_features_cthw = load_packed_tensor(feature_cache_file)
        from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
        raw_features_cthw = bit_label2raw_feature(raw_features_cthw.unsqueeze(0), self.other_args.hbq_round)[0]
        return raw_features_cthw
    
    def prepare_video_input(self, info) -> Tuple:
        filename, begin_frame_id, end_frame_id = (
            info["video_path"],
            info["begin_frame_id"],
            info["end_frame_id"],
        )

        try:
            img_T3HW, raw_features_cthw, feature_cache_file, text_features_lenxdim, text_feature_cache_file = None, None, None, None, None
            img_T3HW_4additional_pn = {}
            text_input = info['caption']
            if '/vdataset/clip' in filename: # clip
                begin_frame_id, end_frame_id = 0, end_frame_id - begin_frame_id
            sample_frames = info['sample_frames']
            tmp_local_path = ''
            if self.use_vae_token_cache:
                feature_cache_file = self.get_video_cache_file(info["video_path"], begin_frame_id, end_frame_id, self.video_fps, info['pn'])
                if osp.exists(feature_cache_file):
                    try:
                        pt = (sample_frames-1) // self.temporal_compress_rate + 1
                        raw_features_cthw = self.load_visual_token(feature_cache_file)
                        assert raw_features_cthw.shape[1] >= pt, f'raw_features_cthw.shape[1] >= pt: {raw_features_cthw.shape[1]} vs {pt}'
                        if raw_features_cthw.shape[1] > pt:
                            raw_features_cthw = raw_features_cthw[:,:pt]
                    except Exception as e:
                        self.print(f'load video cache file error: {e}')
                        os.remove(feature_cache_file)
                        raw_features_cthw = None
                if raw_features_cthw is None and (not self.allow_online_vae_feature_extraction):
                    return False, None
            pn_list = [info['pn']]
            if raw_features_cthw is None:
                tmp_local_path = info["video_path"]
                if not osp.exists(tmp_local_path):
                    return False, None
                video = EncodedVideoDecord(tmp_local_path, os.path.basename(tmp_local_path), num_threads=0)
                start_interval = max(0, begin_frame_id / video._fps)
                end_interval = start_interval+(sample_frames-1)/self.video_fps
                assert end_interval <= video.duration + 0.2, f'{end_interval=}, but {video.duration=}' # 0.2s margin
                end_interval = min(end_interval, video.duration)
                raw_video, _ = video.get_clip(start_interval, end_interval, sample_frames) # rgb order
                h, w, _ = raw_video[0].shape
                h_div_w = h / w
                h_div_w_template = self.h_div_w_templates[np.argmin(np.abs((h_div_w-self.h_div_w_templates)))]
                tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][info['pn']]['pixel']
                    
                for pn in pn_list:
                    img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in raw_video]
                    img_T3HW = torch.stack(img_T3HW, 0)
                    img_T3HW_4additional_pn[pn] = img_T3HW
                del video
                assert img_T3HW.shape[-3:] == (3, tgt_h, tgt_w)
            data_item = {
                'text_input': text_input,
                'img_T3HW': img_T3HW_4additional_pn.get(info['pn'], None),
                'raw_features_cthw': raw_features_cthw,
                'feature_cache_file': feature_cache_file,
                'text_features_lenxdim': text_features_lenxdim,
                'text_feature_cache_file': text_feature_cache_file,
                'meta': info,
            }
            for pn in pn_list[1:]:
                data_item.update({f'img_T3HW_{pn}': img_T3HW_4additional_pn.get(pn, None)})
            return True, data_item
        except Exception as e:
            self.print(f'prepare_video_input error: {e}, info: {info}')
            return False, None

        
    @staticmethod
    def collate_function(batch, online_t5: bool = False) -> None:
        pass
    
    def random_drop_sentences(self, caption, min_sentences):
        elems = [item for item in caption.split('.') if item]
        if len(elems) <= min_sentences:
            return caption
        sentences = self.epoch_rank_generator.integers(min_sentences, len(elems)+1)
        return '.'.join(elems[:sentences]) + '.'

    def __len__(self):
        return len(self.batches) * self.other_args.loop_data_per_epoch

    def get_image_cache_file(self, image_path, pn):
        elems = image_path.split('/')
        elems = [item for item in elems if item]
        filename, ext = osp.splitext(elems[-1])
        filename = get_prompt_id(filename)
        save_filepath = osp.join(self.token_cache_dir, f'images_pn_{pn}', '/'.join(elems[4:-1]), f'{filename}.npz')
        return save_filepath

    def get_video_cache_file(self, video_path, begin_frame_id, end_frame_id, video_fps, pn):
        elems = video_path.split('/')
        elems = [item for item in elems if item]
        filename, ext = osp.splitext(elems[-1])
        filename = get_prompt_id(filename)
        save_filepath = osp.join(self.token_cache_dir, f'pn_{pn}_sample_fps_{video_fps}', '/'.join(elems[4:-1]), f'{filename}_sf_{begin_frame_id}_ef_{end_frame_id}.npz')
        return save_filepath
    