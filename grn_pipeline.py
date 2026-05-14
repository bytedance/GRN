import os
import json
import numpy as np
import torch
from PIL import Image
import tempfile

from grn.utils_t2iv.infer import (
    load_tokenizer,
    load_transformer,
    gen_one_example,
    images2video
)
from grn.utils_t2iv.load import load_visual_tokenizer
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from grn.schedules import get_encode_decode_func


class GRNPipeline:
    def __init__(self, model, vae, text_tokenizer, text_encoder, args, device='cuda'):
        self.model = model
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.args = args
        self.device = device
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = \
            get_encode_decode_func(args.dynamic_scale_schedule)

    @classmethod
    def from_pretrained(
        cls,
        model_path='./weights/model.pth',
        vae_path='./weights/hbq_tokenizer.ckpt',
        text_encoder_ckpt='./weights/umt5-xxl',
        device='cuda',
        torch_dtype=torch.bfloat16,
        hf_repo_id=None,
        task='t2i',
    ):
        # download weights from Hugging Face Hub
        if hf_repo_id:
            from huggingface_hub import hf_hub_download, snapshot_download
            print(f"download weights from Hugging Face Hub: {hf_repo_id}")
            if task == 't2i':
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="GRN_T2I_2B.pth")
            elif task == 't2v':
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="GRN_T2V_2B.pth")
            else:
                raise ValueError(f"Unknown task: {task}")
            vae_path = hf_hub_download(repo_id=hf_repo_id, filename="HBQ_tokenizer_64dim_M4.ckpt")
            snapshot_path = snapshot_download(repo_id=hf_repo_id, allow_patterns="umt5-xxl/**")
            text_encoder_ckpt = os.path.join(snapshot_path, "umt5-xxl")
            print(os.listdir(snapshot_path))
        
        args = cls._get_default_args()
        args.model_path = model_path
        args.vae_path = vae_path
        args.text_encoder_ckpt = text_encoder_ckpt
        if isinstance(device, str):
            device = torch.device(device)
        args.other_device = device
        args.task = task

        # Derived parameters
        args.max_duration = (args.video_frames - 1) / 4
        args.num_of_label_value = args.num_lvl
        args.semantic_num_lvl = args.num_lvl
        args.detail_num_lvl = args.num_lvl
        args.semantic_scale_dim = args.vae_latent_dim
        args.detail_scale_dim = args.vae_latent_dim

        # Load models
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt, device=device)
        vae = load_visual_tokenizer(args, device=device)
        model = load_transformer(vae, args)

        return cls(model, vae, text_tokenizer, text_encoder, args, device)

    @staticmethod
    def _get_default_args():
        class Args:
            def __init__(self):
                self.pn = '1M'
                self.video_frames = 81
                self.model_path = './weights/model.pth'
                self.vae_path = './weights/hbq_tokenizer.ckpt'
                self.text_encoder_ckpt = './weights/umt5-xxl'
                self.cfg = 1
                self.fps = 16
                self.cfg_insertion_layer = 0
                self.vae_latent_dim = 64
                self.hbq_round = 4
                self.rope_type = '3d'
                self.num_lvl = 2
                self.model = 'GRN2b'
                self.rope2d_normalized_by_hw = 2
                self.sampling_per_bits = 1
                self.text_channels = 4096
                self.apply_spatial_patchify = 0
                self.h_div_w_template = 1.0
                self.cache_dir = '/tmp'
                self.checkpoint_type = 'torch'
                self.seed = 42
                self.bf16 = 0
                self.dynamic_scale_schedule = 'GRN_vae_stride16'
                self.train_h_div_w_list = '[]'
                self.max_infer_steps = 50
                self.min_infer_steps = 50
                self.video_caption_type = 'tarsier2_caption'
                self.temporal_compress_rate = 4
                self.cached_video_frames = 81
                self.duration_resolution = 0.25
                self.video_fps = 16
                self.simple_text_proj = 1
                self.min_duration = -1
                self.fsdp_save_flatten_model = 1
                self.use_learnable_dim_proj = 0
                self.use_fsq_cls_head = 1
                self.use_feat_proj = 0
                self.use_clipwise_caption = 0
                self.use_ada_layer_norm = 0
                self.cfg_type = 'cfg_interval_0.0'
                self.add_scale_token = 1
                self.vae_encoder_out_type = 'feature_tanh'
                self.alpha = 1004
                self.refine_mode = 'ar_discrete_GRN_bit'
                self.add_class_token = 0
                self.resample_rand_labels_per_step = 0
                self.cfg_val = 3.0
                self.scale_repetition = ''
                self.gt_leak = -1
                self.use_refined_prompt = None
                self.use_prompt_engineering = 0
                self.quality_prompt = ''
                self.meta = ''
                self.train_split_file = ''
                self.n_sampes = 1
                self.other_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Args()

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.args.other_device = device
        if self.model:
            self.model = self.model.to(device)
        if self.vae:
            self.vae = self.vae.to(device)
        return self

    def __call__(
        self,
        prompt,
        negative_prompt='',
        guidance_scale=3.0,
        temperature=1.0,
        complexity_aware_Tmin=10,
        complexity_aware_Tmax=50,
        complexity_aware_k = 0,
        complexity_aware_b = 50,
        complexity_aware_wp = 5,
        snr_shift = 1.,
        width=512,
        height=512,
        duration=2.,
        generator=None,
        content_type='image',
        seed=None,
        **kwargs
    ):
        if seed is not None:
            self.args.seed = seed
        else:
            self.args.seed = np.random.randint(0, 10000)
        
        self.args.cfg_val = guidance_scale
        self.args.tau = temperature
        self.args.complexity_aware_Tmin = complexity_aware_Tmin
        self.args.complexity_aware_Tmax = complexity_aware_Tmax
        self.args.complexity_aware_k = complexity_aware_k
        self.args.complexity_aware_b = complexity_aware_b
        self.args.complexity_aware_wp = complexity_aware_wp
        self.args.snr_shift = snr_shift
        
        # Get dynamic resolution meta
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
            self.args.dynamic_scale_schedule,
            self.args.train_h_div_w_list,
            self.args.video_frames
        )
        
        # Get scale schedule based on aspect ratio
        h_div_w = height / width if width != 0 else 1.0
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        self.args.mapped_h_div_w_template = h_div_w_template_
        
        if content_type == "image":
            num_frames = 1
            duration = 0
        else:
            num_frames = min(self.args.video_frames, int(duration * self.args.video_fps + 1))
        
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
        
        self.args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        self.args.tower_split_index = self.args.first_full_spatial_size_scale_index + 1
        context_info = self.get_scale_pack_info(scale_schedule, self.args.first_full_spatial_size_scale_index, self.args)
        
        # Generate content
        generated_image = gen_one_example(
            self.model, self.vae, self.text_tokenizer, self.text_encoder, [prompt],
            negative_prompt=negative_prompt, g_seed=seed, gt_leak=self.args.gt_leak, gt_ls_Bl=None,
            cfg_list=self.args.cfg_val, tau_list=self.args.tau, scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer], vae_latent_dim=self.args.vae_latent_dim,
            sampling_per_bits=self.args.sampling_per_bits, enable_positive_prompt=0,
            args=self.args, get_visual_rope_embeds=self.get_visual_rope_embeds,
            context_info=context_info, noise_list=None, class_token_id=0,
        )
        
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        
        generated_image = generated_image.cpu().numpy()
        
        ext = '.jpg' if num_frames == 1 else '.mp4'
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            output_path = tmp.name
        
        images2video(generated_image, fps=self.args.fps, save_filepath=output_path)
        
        if ext == '.jpg':
            img = Image.open(output_path)
            return type('Result', (object,), {'images': [img]})
        else:
            return type('Result', (object,), {'videos': [output_path]})