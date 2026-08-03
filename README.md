<!-- ---
license: mit
title: Generative Refinement Networks
sdk: gradio
emoji: 🚀
colorFrom: red
colorTo: yellow
pinned: true
short_description: "Generative Refinement Networks"
---  -->
# [ECCV 2026] GRN: Generative Refinement Networks

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2604.13030-b31b1b.svg)](https://arxiv.org/abs/2604.13030)
[![Homepage](https://img.shields.io/badge/🏠%20Homepage-GRN-green.svg)](https://bytedance.github.io/GRN/)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue.svg)](https://huggingface.co/bytedance-research/GRN)
[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow.svg)](https://huggingface.co/spaces/hanjian/GRN)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/bytedance/GRN?style=social)](https://github.com/bytedance/GRN)

---

## 🔥 Updates!!
* June 26, 2026: 🏅 GRN 8B is released now! A unified single model covering T2V, I2V and T2I. Its performance rivals Wan 2.1 14B, AR models never surrender!
* June 18, 2026: 🍾 GRN is accepted by ECCV 2026.
* June 8, 2026: ✈️ The training & fine-tuning code for GRN-T2I and GRN-T2V is released.
* June 3, 2026: 🍉 A toy image-video dataset is provided for GRN-T2I/GRN-T2V training and fine-tuning.
* May 23, 2026: 🌺 We release the training and evaluation code for HBQ tokenizer, enjoy~
* April 14, 2026: 🤗 Paper and code release

## 📋 Table of Contents

- [🌟 Introduction](#-introduction)
- [✨ Gallery](#-gallery)
- [🚀 Demo](#-demo)
- [📦 Model Zoo](#-model-zoo)
- [🛠️ Installation](#️-installation)
- [📦 HBQ Tokenizer](#-hbq-tokenizer)
  - [Data](#data)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [🖼️ Class-to-Image](#️-class-to-image)
  - [Data](#data-1)
  - [Training](#training-2)
  - [Evaluation](#evaluation-1)
- [🎨 Text-to-Image](#-text-to-image)
  - [Data](#data-2)
  - [Training](#training-2)
  - [Inference](#inference)
- [🎬 Text-to-Video](#-text-to-video)
  - [Data](#data-3)
  - [Training](#training-3)
  - [Inference](#inference-1)
- [🎬 Image-to-Video](#-image-to-video)
  - [Data](#data-4)
  - [Training](#training-4)
  - [Inference](#inference-2)
- [📧 Contact](#-contact)
- [🤗 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)

---

## 🌟 Introduction

This is the official implementation of the paper **Generative Refinement Networks for Visual Synthesis**. Neither diffusion nor autoregressive — GRN is a third way. 🧠 Refines globally like an artist. ⚡ Generates adaptively by complexity. 🏆 New SOTA across image & video. The visual generation paradigm just got rewritten.

Diffusion models dominate visual generation but they allocate uniform computational effort to samples with varying levels of complexity. Autoregressive (AR) models are complexity-aware, as evidenced by their variable likelihoods, but suffer from lossy tokenization and error accumulation.

We introduce **Generative Refinement Networks (GRN)**, a new visual synthesis paradigm that addresses these issues:
- **Near-lossless tokenization** via Hierarchical Binary Quantization (HBQ)
- **Global refinement mechanism** that progressively perfects outputs like a human artist
- **Entropy-guided sampling** for complexity-aware, adaptive-step generation

GRN achieves state-of-the-art results on ImageNet reconstruction and class-conditional generation, and scales effectively to text-to-image and text-to-video tasks.

---

<figure align="center">
  <figcaption><strong><em>Generative Refinement Framework</em></strong></figcaption>
  <img src="assets/framework.jpg" width="100%" alt="Framework">
</figure>

<p align="center">
Starting from a random token map, GRN randomly selects more predictions at each step and refines all input tokens. For example, compared to the second step, the third step filled six new tokens (<span style="color: rgb(220, 120, 117);">pink</span>), kept two tokens (<span style="color: rgb(88, 160, 227);">blue</span>), erased two tokens (<span style="color: rgb(240, 180, 40);">yellow</span>), and left six tokens blank (<span style="color: rgb(128, 138, 151);">gray</span>).
</p>

---

## ✨ Gallery

### GRN-8B Text-to-Video Examples

<div align="center">
  <table style="border-spacing: 6px; margin: auto;">
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/6ce844dc-3185-4239-bcf1-d72ff20a3031" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/1697066e-f00f-4e23-a55c-c6af5948c4af" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/1023ea4d-d814-4be1-95f2-b1623de0f6bd" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/6244dae4-f480-408a-ac3d-19e4d1ef0a2d" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/5aefc8d2-bc99-48e4-bd1c-9b3077c9c35e" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/014e8bb4-04a7-4fa4-a597-d0dfbcc23e02" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/6bde1f2e-cebe-4f47-9eac-4fe817c3ebc7" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/b9957300-fa98-411c-83d5-f972621245ad" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/d07cef92-3eec-4e7c-93da-f6c6a8dc1658" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
  </table>
</div>

---

### GRN-8B Image-to-Video Examples

<div align="center">
  <table style="border-spacing: 6px; margin: auto;">
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/527f94b0-4b04-4cbb-a86d-9f5ae05fab67" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/0b63a9ed-2940-402f-8339-db0d05a09525" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/8d108a5f-1414-43ca-af6b-8862640741e5" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/64cd45a9-0c2f-4926-bcc0-b8a0a939ae54" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/6c31c9e5-0742-4416-925c-16c39bc5a03a" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/4e966b46-6107-4ffe-a24b-37dc3c8461dd" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
    <tr>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/56ce2dc1-3b64-4493-ab27-b2ba273c64ef" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/ef45a3f4-8fb2-4bb5-885e-19645e5a0fb5" width="33%" autoplay muted loop playsinline></video></td>
      <td style="padding: 2px;"><video src="https://github.com/user-attachments/assets/98e3fbb0-9a54-49e6-8cec-96a42d0634e6" width="33%" autoplay muted loop playsinline></video></td>
    </tr>
  </table>
</div>

### GRN-2B Class-to-Image Examples
<figure align="center">
  <!-- <figcaption><strong><em>GRN-2B Class-to-Image Examples</em></strong></figcaption> -->
  <img src="assets/c2i_examples.jpg" width="100%" alt="Class-to-Image Examples">
</figure>

### GRN-2B Text-to-Image Examples
<figure align="center">
  <!-- <figcaption><strong><em>GRN-2B Text-to-Image Examples</em></strong></figcaption> -->
  <img src="assets/t2i_examples.jpg" width="100%" alt="Text-to-Image Examples">
</figure>

---

## 🚀 Demo

### 🖼️ Text-to-Image
Try our interactive Text-to-Image demo on 🤗 Hugging Face Space:

**[GRN T2I Demo](https://huggingface.co/spaces/hanjian/GRN)**

Experience the power of Generative Refinement Networks firsthand by generating images from text prompts directly in your browser!

---

### 🎬 Text-to-Video
Try our interactive Text-to-Video demo on Discord:

[![Discord](https://img.shields.io/badge/Discord-Join%20Server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](http://opensource.bytedance.com/discord/invite)


<figure align="center">
  <figcaption><strong><em>T2V Demo on Discord</em></strong></figcaption>
  <img src="assets/t2v_demo.png" width="100%" alt="T2V Demo">
</figure>

---

## 📦 Model Zoo

| Model | Checkpoints |
|-------|:-----------:|
| **Tokenizers** | ✅ [ImageNet Tokenizer](https://huggingface.co/bytedance-research/GRN/blob/main/HBQ_image_tokenizer_16dim_M4.ckpt)<br>✅ [Joint Image/Video Tokenizer](https://huggingface.co/bytedance-research/GRN/blob/main/HBQ_tokenizer_64dim_M4.ckpt) |
| **GRN_ind_C2I** | ✅ [B](https://huggingface.co/bytedance-research/GRN/blob/main/GRN_ind_B_ep599.pth)<br>[L](https://huggingface.co/bytedance-research/GRN/blob/main/GRN_ind_B_ep599.pth)<br>[H](https://huggingface.co/bytedance-research/GRN/blob/main/GRN_ind_B_ep599.pth)|
| **GRN_bit_T2I** | ✅ [GRN_T2I](https://huggingface.co/bytedance-research/GRN/blob/main/GRN_T2I_2B.pth) |
| **GRN_bit_T2V** | ✅ [GRN_T2V](https://huggingface.co/bytedance-research/GRN/blob/main/GRN_T2V_2B.pth) |

---

## 🛠️ Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/bytedance/GRN
cd GRN
```

### Step 2: Create conda environment
A suitable [conda](https://conda.io/) environment named `GRN` can be created and activated with:
```bash
conda create -n GRN python=3.11
conda activate GRN
pip install -r requirements.txt
```

### Troubleshooting
If you get `undefined symbol: iJIT_NotifyEvent` when importing `torch`, simply:
```bash
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.

---

## 📦 HBQ Tokenizer

### Data
Image Dataset, e.g., data_root/username/labels/imagenet/train.txt:
```
[image_1_full_path]
[image_2_full_path]
[image_3_full_path]
...
```

Video Dataset, e.g., data_root/username/labels_hanjian/high-quality-video/horizontal_videos.txt
```
[video_1_full_path]
[video_2_full_path]
[video_3_full_path]
...
```

### Training
For example, set `latent_channels=16/64` and `quant_method=hierarchical_binary_quant_round_4` in `scripts/hbq_tokenizer_train.sh`, then run:
```bash
cd grn/tokenizer
bash scripts/hbq_tokenizer_train.sh
```

### Evaluation
For example, set `latent_channels=16/64` and `quant_method=hierarchical_binary_quant_round_4` in `scripts/hbq_tokenizer_train.sh`, then run:
```bash
cd grn/tokenizer
bash scripts/hbq_tokenizer_eval.sh
```

---

## 🖼️ Class-to-Image

### Data
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Training

All training scripts are located in `scripts/c2i/`. We suggest using 8x80GB GPUs for most models.

| Model | Training Script | GPUs Required |
|-------|:-------------:|:-------------:|
| GRN_ind_B | `bash scripts/c2i/train_GRN_ind_B.sh` | 8x80GB |
| GRN_bit_B | `bash scripts/c2i/train_GRN_bit_B.sh` | 8x80GB |
| GRN_ind_L | `bash scripts/c2i/train_GRN_ind_L.sh` | 8x80GB |
| GRN_ind_H | `bash scripts/c2i/train_GRN_ind_H.sh` | 16x80GB |
| GRN_ind_G | `bash scripts/c2i/train_GRN_ind_G.sh` | 32x80GB |

### Evaluation

PyTorch pre-trained models are available [here](https://huggingface.co/bytedance-research/GRN/tree/main).

All evaluation scripts are located in `scripts/c2i/`. We suggest using 8x80GB vRAM GPUs.

| Model | Evaluation Script |
|-------|:--------------:|
| GRN_ind_B | `bash scripts/c2i/eval_GRN_ind_B.sh` |
| GRN_bit_B | `bash scripts/c2i/eval_GRN_bit_B.sh` |
| GRN_ind_L | `bash scripts/c2i/eval_GRN_ind_L.sh` |
| GRN_ind_H | `bash scripts/c2i/eval_GRN_ind_H.sh` |
| GRN_ind_G | `bash scripts/c2i/eval_GRN_ind_G.sh` |

We use [torch-fidelity](https://github.com/LTH14/torch-fidelity) to evaluate FID and IS against a reference image folder or statistics. We use the JiT's pre-computed reference stats under `grn/utils_c2i/fid_stats`.

---

## 🎨 Text-to-Image
### Data
Refer to `data/toy_data/jsonls/000001/0001_0800_000000100.jsonl`
```
{"image_path": "[image_path_1]", "long_caption": "xxx", "long_caption_type": "caption-InternVL2.0", "text": "", "short_caption_type": "blip2_caption", "width": 1080, "height": 1920}
{"image_path": "[image_path_2]", "long_caption": "xxx", "long_caption_type": "caption-InternVL2.0", "text": "", "short_caption_type": "blip2_caption", "width": 1080, "height": 1920}
...
```

### Training
Run `bash scripts/t2iv/train_GRN_bit_t2iv.sh`

### Inference

You can simply run `python3 tools/t2i_infer.py` or use the following code:

```python
from PIL import Image
from tools.grn_pipeline import GRNPipeline

# Load pipeline
pipeline = GRNPipeline.from_pretrained(
    hf_repo_id='bytedance-research/GRN',
    task='T2I',
    pn='1M', 
    model='GRN2b',
    device='cpu',
).to('cuda')

# Generate one image
result = pipeline(
    prompt="<T2I>" + "A cute cat playing in the garden",
    guidance_scale=3.0,
    temperature=1.1,
    complexity_aware_Tmin=10,
    complexity_aware_Tmax=50,
    complexity_aware_k = 0,
    complexity_aware_b = 50,
    complexity_aware_wp = 5,
    snr_shift = 1.,
    h_div_w=1.,
    content_type='image',
    seed=42,
)
image = result.images[0]
image.save('./generated_image.jpg')
```

---

## 🎬 Text-to-Video
### Data
Refer to `data/toy_data/jsonls/000001/0001_0800_000000100.jsonl`
```
{"video_path": "[video_path_1]", "begin_frame_id": xxx, "end_frame_id": xxx, "quality_prompt": "There is text in the video.", "fps": 25.0, "duration": 3.88, "width": 1280, "height": 720, "caption": [{"type": "short", "content": "[short_caption]"}, {"type": "medium", "content": "[medium_caption]"}, {"type": "long", "content": "[long_caption]"}]}
{"video_path": "[video_path_1]", "begin_frame_id": xxx, "end_frame_id": xxx, "quality_prompt": "The quality is very high!", "fps": 25.0, "duration": 3.88, "width": 1280, "height": 720, "caption": [{"type": "short", "content": "[short_caption]"}, {"type": "medium", "content": "[medium_caption]"}, {"type": "long", "content": "[long_caption]"}]}
...
```

### Training
Run `bash scripts/t2iv/train_GRN_bit_t2iv.sh`

### Inference

You can simply run `python3 tools/t2v_infer.py` or use the following code:

```python
from tools.grn_pipeline import GRNPipeline

# Load pipeline
pipeline = GRNPipeline.from_pretrained(
    hf_repo_id='bytedance-research/GRN',
    task='T2V',
    pn='0.41M', 
    model='GRN2b', # 'GRN2b' or 'GRN8b'
    device='cpu',
).to('cuda')

# Generate one video
result = pipeline(
    prompt="Two women demonstrate a makeup product, applying it with a sponge while smiling and engaging with the camera in a bright, clean setting.",
    guidance_scale=4.0,
    temperature=1.0,
    complexity_aware_Tmin=10,
    complexity_aware_Tmax=50,
    complexity_aware_k = 0,
    complexity_aware_b = 50,
    complexity_aware_wp = 5,
    snr_shift = 1.,
    h_div_w=9/16,
    duration=2.,
    content_type='video',
    seed=42,
)
video_file = result.videos[0]
```

---
## 🎭 Image-to-Video
### Data
Refer to `data/toy_data/jsonls/000001/0001_0800_000000100.jsonl`
```
{"video_path": "[video_path_1]", "begin_frame_id": xxx, "end_frame_id": xxx, "quality_prompt": "There is text in the video.", "fps": 25.0, "duration": 3.88, "width": 1280, "height": 720, "caption": [{"type": "short", "content": "[short_caption]"}, {"type": "medium", "content": "[medium_caption]"}, {"type": "long", "content": "[long_caption]"}]}
{"video_path": "[video_path_1]", "begin_frame_id": xxx, "end_frame_id": xxx, "quality_prompt": "The quality is very high!", "fps": 25.0, "duration": 3.88, "width": 1280, "height": 720, "caption": [{"type": "short", "content": "[short_caption]"}, {"type": "medium", "content": "[medium_caption]"}, {"type": "long", "content": "[long_caption]"}]}
...
```

### Training
Run `bash scripts/t2iv/train_GRN_bit_t2iv.sh`

### Inference

You can simply run `python3 tools/i2v_infer.py` or use the following code:

```python
from tools.grn_pipeline import GRNPipeline

# Load pipeline
pipeline = GRNPipeline.from_pretrained(
    hf_repo_id='bytedance-research/GRN',
    task='T2V',
    pn='0.41M', 
    model='GRN8b',
    device='cpu',
).to('cuda')

# Generate one video
result = pipeline(
    prompt="<I2V>视频展示了一辆红色敞篷跑车在城市道路中行驶的连续画面。车辆以中等速度前进，车身光滑，反射着黄昏的暖光，黑色轮毂与红色车漆形成对比。驾驶员为男性，专注地操控方向盘，姿态放松。道路两侧排列着高大的棕榈树，背景中可见石质围栏和模糊的建筑轮廓。随着视频推进，一辆白色SUV从后方快速驶过，产生动态模糊，突显跑车的稳定行驶。镜头保持相对固定的侧前方视角，轻微跟随车辆移动，捕捉车身线条与光影变化。整体画面色调温暖，光线柔和，营造出一种优雅而动感的都市驾驶氛围. high aesthetic and high quality video.",
    guidance_scale=4.0,
    temperature=1.0,
    complexity_aware_Tmin=10,
    complexity_aware_Tmax=50,
    complexity_aware_k = 0,
    complexity_aware_b = 50,
    complexity_aware_wp = 5,
    snr_shift = 1.,
    h_div_w=9/16,
    duration=2.,
    content_type='video',
    seed=42,
    first_frame_condition=True,
    first_frame_path='./assets/i2v_example.jpg',
)
video_file = result.videos[0]
```

---

## 📧 Contact

If you are interested in scaling GRN for image generation / image editing / video generation / video editing / unified model directions, please feel free to reach out!

**📧 Email:** [hanjian.thu123@bytedance.com](mailto:hanjian.thu123@bytedance.com)

---

## 🤗 Acknowledgements

- Thanks to [JiT](https://github.com/LTH14/JiT), [Infinity](https://github.com/FoundationVision/Infinity) and [InfinityStar](https://github.com/FoundationVision/InfinityStar) for their wonderful work and codebase!

---

## 📝 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{han2026grn,
      title={Generative Refinement Networks for Visual Synthesis}, 
      author={Jian Han and Jinlai Liu and Jiahuan Wang and Bingyue Peng and Zehuan Yuan},
      year={2026},
      eprint={2604.13030},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.13030}, 
}
```
