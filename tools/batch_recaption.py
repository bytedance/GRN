import os
import torch
from PIL import Image
import cv2
import json
import glob
import argparse
import numpy as np
from typing import List, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os.path as osp

from grn.dataset.dataset_joint_vi import local_or_download
from grn.utils.video_decoder import EncodedVideoDecord



# --- 模型加载 (建议使用 try-except 保证健壮性) ---
try:
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    # model_id = "/tmp/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, padding_side='left')
except Exception as e:
    print(f"模型加载失败: {e}")
    # 在实际生产环境中，可能需要更复杂的错误处理或退出逻辑
    exit()

FRAMES_PER_SECOND = 1
CN_PROMPT = """
你正在为一个文本生成视频（text-to-video）训练数据集生成视频描述。
1、观看视频，并生成三种不同粒度的描述：short、medium 和 long。
2、这些描述应基于视频中可见的内容，保持客观、准确且具备描述性。它们应捕捉视频生成模型需要学习的信息，包括主体、动作、运动、场景、空间布局、镜头运动、风格和时间推进过程。
3、不要幻觉生成内容。不要推断隐藏的意图、身份、地点、品牌或故事。只描述视频中可见的证据。

输出格式（每段描述必须另起一行，并以指定标题开头）：
[short]: 描述主要主体、动作和场景，不超过 20 个词。
[medium]: 描述主要主体、动作、环境，以及重要的视觉或镜头细节，长度为 40 到 60 个词。
[long]: 描述视频完整的时间推进过程，包括主体细节、动作、互动、场景布局、镜头运动、以及随时间发生的变化，长度为 80 到 130 个词。
""".strip()

EN_PROMPT="""
You are generating captions for a text-to-video training dataset.
1. Watch the video and produce three captions with different granularity: short, medium, and long.
2. The captions should be visually grounded, objective, and descriptive. They should capture what a video generation model needs to learn: subject, action, motion, scene, spatial layout, camera movement, style, and temporal progression.
3. Do not hallucinate. Do not infer hidden intent, identity, location, brand, or story. Describe only visible evidence.

Output format (each description must start on a new line with the specified header):
[short]: describe the main subject, action, and scene, in no more than 20 words.
[medium]: describe the main subject, action, environment, and important visual or camera details, in 40 to 60 words.
[long]: describe the full temporal progression of the video, including subject details, actions, interactions, scene layout, camera motion, and changes over time, in 80 to 130 words.
""".strip()

COMBINED_PROMPT = [CN_PROMPT, EN_PROMPT]

def extract_frames(video_path: str, fps: int) -> List[Image.Image]:
    """从视频中按指定的帧率提取帧 (基本保持不变，但增加了错误处理)"""
    if not os.path.exists(video_path):
        # print(f"警告：找不到视频文件 '{video_path}'，跳过处理。")
        return []

    frames = []
    try:
        video_capture = cv2.VideoCapture(video_path)
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        if video_fps == 0: # 视频文件可能已损坏
            return []

        frame_interval = int(video_fps / fps)
        if frame_interval == 0:
            frame_interval = 1

        frame_count = 0
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            
            if frame_count % frame_interval == 0:
                h, w, _ = frame.shape
                # 统一缩放逻辑
                if h > w:
                    new_w = 480
                    new_h = int(new_w * h / w)
                else:
                    new_h = 480
                    new_w = int(new_h * w / h)
                
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            frame_count += 1
    except Exception as e:
        print(f"处理视频 '{video_path}' 时出错: {e}")
        import pdb; pdb.set_trace()
        return []
    finally:
        if 'video_capture' in locals() and video_capture.isOpened():
            video_capture.release()
    
    return frames

import re
from typing import Dict
def parse_combined_caption(text: str) -> Dict[str, str]:
    """
    从模型生成的合并文本中解析出三种描述。
    - short: [short] 和 [medium] 之间的内容
    - medium: [medium] 和 [long] 之间的内容
    - long: [long] 之后的所有内容
    """
    captions = {'short': '', 'medium': '', 'long': ''}
    # 为了方便处理，统一文本格式，将所有[tag]:替换为[tag]
    text = text.replace('[short]:', '[short]').replace('[medium]:', '[medium]').replace('[long]:', '[long]')
    # 使用正则表达式查找各个部分的文本
    # re.DOTALL (或 re.S) 标志让 '.' 可以匹配包括换行符在内的任意字符
    short_match = re.search(r'\[short\](.*?)\[medium\]', text, re.DOTALL)
    if short_match:
        # group(1) 获取第一个捕获组的内容
        captions['short'] = short_match.group(1).strip(' :\n\r\t')
    medium_match = re.search(r'\[medium\](.*?)\[long\]', text, re.DOTALL)
    if medium_match:
        captions['medium'] = medium_match.group(1).strip(' :\n\r\t')
    long_match = re.search(r'\[long\](.*)', text, re.DOTALL)
    if long_match:
        captions['long'] = long_match.group(1).strip(' :\n\r\t')
    return captions


class VideoDataset(Dataset):
    def __init__(self, metas: List[Dict], fps: int):
        self.metas = metas
        self.fps = fps

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        # video_path = meta['video_path']
        video_path = local_or_download(meta, 500)
        if not osp.exists(video_path):
            return {'meta': meta, 'frames': None}

        exists_frame_id = 'begin_frame_id' in meta
        if exists_frame_id:
            begin_frame_id, end_frame_id = meta["begin_frame_id"], meta["end_frame_id"]
            if '/vdataset/clip' in meta['video_path']: # clip
                begin_frame_id, end_frame_id = 0, end_frame_id - begin_frame_id
        try:
            video = EncodedVideoDecord(video_path, os.path.basename(video_path), num_threads=0) # bgr
            if exists_frame_id:
                start_interval = max(0, begin_frame_id / video._fps)
                end_interval = end_frame_id / video._fps
            else:
                start_interval = 0
                end_interval = video._duration
                begin_frame_id = 0
                end_frame_id = end_interval * video._fps
                meta['begin_frame_id'] = begin_frame_id
                meta['end_frame_id'] = end_frame_id
                meta['fps'] = video._fps
                meta['duration'] = video._duration

            sample_frames = int(np.round((end_interval - start_interval))) + 1
            frames, _ = video.get_clip(start_interval, end_interval, sample_frames)
            h, w, _ = frames[0].shape
            meta['width'] = w
            meta['height'] = h
            
            if h > w:
                new_w = 480
                new_h = int(new_w * h / w)
            else:
                new_h = 480
                new_w = int(new_h * w / h)
            frames = [Image.fromarray(cv2.resize(frame, (new_w, new_h))).convert("RGB") for frame in frames] # bgr to rgb
        except Exception as e:
            frames = None
            print(e)
        if osp.exists(video_path):
            os.remove(video_path)
        # frames = extract_frames(video_path, self.fps)
        if not frames:
            return {'meta': meta, 'frames': None}
        return {'meta': meta, 'frames': frames}

def collate_fn(batch: List[Dict[str, Any]]):
    """
    自定义的 collate_fn 用于处理可变数量的帧和可能的空数据。
    """
    # 过滤掉帧提取失败的样本
    valid_samples = [item for item in batch if item['frames'] is not None]
    if not valid_samples:
        return None

    # 准备模型输入
    batch_inputs = []
    for item in valid_samples:
        query = [{"type": "image", 'image': frame} for frame in item['frames']]
        query.append({"type": "text", 'text': np.random.choice(COMBINED_PROMPT)})
        messages = [{"role": "user", "content": query}]
        batch_inputs.append(messages)

    # 使用 processor 进行批处理
    inputs = processor.apply_chat_template(
        batch_inputs,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True # 关键：启用 padding 以处理不同长度的序列
    )
    
    # 将原始元数据也传递下去
    metas = [item['meta'] for item in valid_samples]
    return {'inputs': inputs, 'metas': metas}

def main():
    # --- 配置参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl_dir', type=str, default='./data/alive_sft_data/sd_pixverse/full')
    parser.add_argument('--output_jsonl_dir', type=str, default='./data/alive_sft_data/sd_pixverse/full_recaption')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    jsonl_files = glob.glob(osp.join(args.input_jsonl_dir, '*/*.jsonl'))
    np.random.shuffle(jsonl_files)

    for jsonl_file in jsonl_files:
        print(f"\n--- 正在处理文件: {jsonl_file} ---")
        try:
            save_file = jsonl_file.replace(args.input_jsonl_dir, args.output_jsonl_dir)
            
            # 优化点 4: 先读取所有已处理的记录，避免文件I/O
            processed_videos = set()
            if osp.exists(save_file):
                with open(save_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            meta = json.loads(line)
                            processed_videos.add(meta['video_path'])
                        except json.JSONDecodeError:
                            continue # 跳过损坏的行
                print(f"已在 {save_file} 中找到 {len(processed_videos)} 条已处理记录。")

            # 读取需要处理的元数据
            metas_to_process = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        meta = json.loads(line)
                        if 'duration' in meta and meta['duration'] > 13:
                            continue
                        if meta['video_path'] not in processed_videos:
                            metas_to_process.append(meta)
                    except json.JSONDecodeError:
                        continue
            
            if not metas_to_process:
                print("此文件中的所有视频均已处理，跳过。")
                continue

            print(f"需要处理 {len(metas_to_process)} 个新视频。")

            # 创建 Dataset 和 DataLoader
            dataset = VideoDataset(metas_to_process, FRAMES_PER_SECOND)
            dataloader = DataLoader(
                dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, # 通常推理不需要打乱
                num_workers=NUM_WORKERS, 
                collate_fn=collate_fn,
                pin_memory=True # 如果使用GPU，可以加速数据传输
            )

            newly_processed_lines = []
            # 使用 tqdm 显示进度条
            for batch in tqdm(dataloader, desc="处理批次"):
                if batch is None:
                    continue
                inputs = batch['inputs'].to(model.device)
                metas = batch['metas']

                # 一次性为整个批次生成结果
                generated_ids = model.generate(**inputs, max_new_tokens=400) # 增加 token 预算以容纳三种描述
                
                # 解码时也需要注意与输入ID的对齐
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # 后处理并准备写入
                for i, text in enumerate(output_texts):
                    meta = metas[i]
                    parsed_captions = parse_combined_caption(text)
                    
                    if 'tarsier2_caption' in meta:
                        del meta['tarsier2_caption']
                    
                    if 'MiniCPM_V_2_6_caption' in meta:
                        del meta['MiniCPM_V_2_6_caption']
                    
                    meta['caption'] = []
                    short = parsed_captions.get('short', '')
                    medium = parsed_captions.get('medium', '')
                    long = parsed_captions.get('long', '')
                    if short:
                        meta['caption'].append({'type': 'short', 'content': short})
                    if medium:
                        meta['caption'].append({'type': 'medium', 'content': medium})
                    if long:
                        meta['caption'].append({'type': 'long', 'content': long})
                    if len(meta['caption']):
                        newly_processed_lines.append(json.dumps(meta, ensure_ascii=False) + '\n')

                if newly_processed_lines:
                    os.makedirs(osp.dirname(save_file), exist_ok=True)
                    # 使用 'a' (append) 模式，将新处理的结果追加到文件中
                    with open(save_file, 'a', encoding='utf-8') as f:
                        f.writelines(newly_processed_lines)
                    print(f"成功向 {save_file} 追加了 {len(newly_processed_lines)} 条新记录。")
                    newly_processed_lines = []

        except Exception as e:
            print(f"处理文件 {jsonl_file} 时发生严重错误: {e}")

if __name__ == "__main__":
    main()
