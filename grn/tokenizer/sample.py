import os
import tqdm
import json
import re
import torch
import torch.nn.functional as F
import argparse
import time
import datetime
import numpy as np
import hashlib
import random
import torch.nn as nn
from torchvision.models.inception import inception_v3
from torch.profiler import record_function as torch_record_function
from contextlib import nullcontext
import lpips
import cv2
from einops import rearrange
from tqdm import tqdm
from PIL import Image
import os.path as osp
Image.MAX_IMAGE_PIXELS = None

from videovae.modules.commitments import DiagonalGaussianDistribution

import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP

import imageio
import random
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from videovae.data import VideoData
from videovae.utils.misc import save_video_grid, shift_dim, data_prefix_manager, rearranged_forward, seed_everything
from videovae.utils.init_models import init_cnn_from_image, load_cnn
from videovae.utils.arguments import MainArgs, add_model_specific_args, init_resolution
from videovae.evaluation import get_fvd_logits, frechet_distance, load_fvd_model
from videovae.evaluation import calculate_frechet_distance
from videovae.evaluation import InceptionV3
from videovae.evaluation import calculate_fvd, calculate_lpips, calculate_psnr, calculate_ssim

torch.set_num_threads(32)
os.environ["NCCL_DEBUG"] = "WARN"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def calculate_batch_codebook_usage_percentage(batch_encoding_indices,n_codes):
    if isinstance(batch_encoding_indices, list):
        all_indices = []
        for one_encoding_indices in batch_encoding_indices:
            all_indices.append(one_encoding_indices.flatten())
        all_indices = torch.cat(all_indices, dim=0)
    else:
        # Flatten the batch of encoding indices into a single 1D tensor
        all_indices = batch_encoding_indices.flatten()
    all_indices = all_indices.detach().cpu()
    
    # Obtain the total number of encoding indices in the batch to calculate percentages
    total_indices = all_indices.numel()
    
    # Initialize a tensor to store the percentage usage of each code
    codebook_usage = torch.zeros(n_codes, dtype=torch.long)
    
    # Count the number of occurrences of each index and get their frequency as percentages
    unique_indices, counts = torch.unique(all_indices, return_counts=True)
    
    # Populate the corresponding percentages in the codebook_usage_percentage tensor
    codebook_usage[unique_indices.long()] = counts
    
    return codebook_usage


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def default_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqgan_ckpt', type=str, default=None)
    parser.add_argument('--sd_ckpt', type=str, default=None)
    parser.add_argument('--use_frames', type=int, default=None)
    parser.add_argument('--inference_type', type=str, choices=["image", "video", "video_concat"])
    parser.add_argument('--save_prediction', action='store_true')
    parser.add_argument('--save_dir',  type=str, default="results")
    parser.add_argument('--intermediate_tensor', action='store_true')
    parser.add_argument('--save_z', action='store_true')
    parser.add_argument('--save_frames', action='store_true')
    parser.add_argument('--image_recon4video', action='store_true')
    parser.add_argument('--junke_old', action='store_true')
    parser.add_argument('--cal_norm', action='store_true')
    parser.add_argument('--save_samples', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser = MainArgs.add_main_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args, parser, vae_model = add_model_specific_args(args, parser)
    args = parser.parse_args()
    return args, vae_model


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355+int(time.time())%1000)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30 * 60))

def cleanup():
    dist.destroy_process_group()

def main():
    args, vae_model = default_parse_args()
    assert len(args.dataset_list) == 1

    # init data_prefix_manager
    data_prefix_manager.set_data_root(args.data_root, username=args.username)
    args.default_root_dir = data_prefix_manager(args.default_root_dir)
    os.makedirs(args.default_root_dir, exist_ok=True)
    print(args.default_root_dir)

    # init intermediate_tensor_dir
    if args.intermediate_tensor:
        random.seed(time.time())
        random_folder_name = hashlib.sha256(str(random.random()).encode('utf-8')).hexdigest()[:16]
        args.intermediate_tensor_dir = os.path.join(args.default_root_dir, random_folder_name)
        print(f"save temporal tensor to {args.intermediate_tensor_dir}")
    
    seed_everything(seed=0, allow_tf32=True) # ALERT: allow_tf32=True may cause accumulate error in conv3d forward > 

    # init resolution
    args.resolution = init_resolution(args.resolution, len(args.dataset_list))
 
    # init profiler
    def trace_handler(p):
        p.export_chrome_trace(os.path.join(args.default_root_dir, f"trace_step_{p.step_num}_rank_{0}.json"))

    tp = None
    if args.turn_on_profiler:
        tp = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profiler_scheduler_wait_steps,
                warmup=3,
                active=2,
                repeat=1,
            ),
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=trace_handler
        )
        tp.start()
        record_function = torch_record_function
    else:
        record_function = nullcontext


    vae = None
    use_vae = None
    num_codes = None
    if args.vqgan_ckpt:
        args.vqgan_ckpt = data_prefix_manager(args.vqgan_ckpt)
    if args.tokenizer in ["hbq_tokenizer"]:
        vae = vae_model(args)
        state_dict = torch.load(args.vqgan_ckpt, map_location=torch.device("cpu"), weights_only=True)
        new_state_dict = {}
        for key in ['vae', 'ema']:
            if (key not in state_dict) or (not state_dict[key]):
                continue
            if 'quantizer.scale_learnable_parameters' in state_dict[key]:
                if len(state_dict[key]['quantizer.scale_learnable_parameters']) == 1:
                    state_dict[key]['quantizer.scale_learnable_parameters'] = state_dict[key]['quantizer.scale_learnable_parameters'].expand(4)
                state_dict[key]['scale_learnable_parameters'] = state_dict[key]['quantizer.scale_learnable_parameters']
                del state_dict[key]['quantizer.scale_learnable_parameters']
            if 'z_mean' in state_dict[key]:
                if state_dict[key]['z_mean'].shape != vae.z_mean.shape:
                    del state_dict[key]['z_mean']
                    del state_dict[key]['z_std']
            new_state_dict[key] = state_dict[key]
            slim_model_path = args.vqgan_ckpt.replace('/checkpoints/', f'/slim_{key}/')
            if not osp.exists(slim_model_path):
                os.makedirs(os.path.dirname(slim_model_path), exist_ok=True)
                torch.save({key: state_dict[key]}, slim_model_path)
                print(f'save to {slim_model_path}')

        if args.ema == "yes":
            print("testing ema weights")
            print(vae.load_state_dict(new_state_dict["ema"], strict=False))
        else:
            print("testing non ema weights")
            print(vae.load_state_dict(new_state_dict["vae"], strict=False))
        for name, param in vae.named_parameters():
            if name.startswith("scale_learnable_"):
                try:
                    print(f"{name}: {param[:32,0,0].cpu().detach().reshape(-1).tolist()}")
                except:
                    print(f"{name}: {param[:32].cpu().detach().reshape(-1).tolist()}")
        for name, param in vae.named_buffers():
            if name.startswith("scale_learnable_"):
                try:
                    print(f"{name}: {param[:32,0,0].cpu().detach().reshape(-1).tolist()}")
                except:
                    print(f"{name}: {param[:32].cpu().detach().reshape(-1).tolist()}")
            if ("scale_wise_std_" in name) or ("scale_wise_mean_" in name):
                print(f"{name}: {param[:32,0,0].cpu().detach().reshape(-1).tolist()}")
            if ('signal_' in name):
                print(f"{name}: {param.cpu().detach().reshape(-1).tolist()}")
        if args.tokenizer != 'hbq_tokenizer':
            vae.enable_slicing()
        # vae.enable_tiling()
    else:
        raise NotImplementedError

    if args.inference_type == "video":
        def extract_results(return_dict, world_size):
            real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs = [], [], [], [], []
            if args.intermediate_tensor:
                for rank in range(world_size):
                    real_embeddings.append(return_dict[rank]['real_embeddings'])
                    fake_embeddings.append(return_dict[rank]['fake_embeddings'])
                    all_real_videos += return_dict[rank]['all_real_videos']
                    all_fake_videos += return_dict[rank]['all_fake_videos']
                    zs.append(return_dict[rank]['zs'])
                real_embeddings = torch.cat(real_embeddings, 0).to('cuda:0')
                fake_embeddings = torch.cat(fake_embeddings, 0).to('cuda:0')
                zs = torch.cat(zs, 0).to('cuda:0')
            else:
                for rank in range(world_size):
                    real_embeddings.append(return_dict[rank]['real_embeddings'])
                    fake_embeddings.append(return_dict[rank]['fake_embeddings'])
                    all_real_videos.append(return_dict[rank]['all_real_videos'])
                    all_fake_videos.append(return_dict[rank]['all_fake_videos'])
                    zs.append(return_dict[rank]['zs'])
                real_embeddings = torch.cat(real_embeddings, 0).to('cuda:0')
                fake_embeddings = torch.cat(fake_embeddings, 0).to('cuda:0')
                all_real_videos = torch.cat(all_real_videos, 0)
                all_fake_videos = torch.cat(all_fake_videos, 0)
                zs = torch.cat(zs, 0).to('cuda:0')
            return real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs

        def inference(mean=None, std=None, noise_scale=0):
            world_size = torch.cuda.device_count()
            manager = torch.multiprocessing.Manager()
            return_dict = manager.dict()
            ### multi-process
            # try:
            #     spawn(inference_DDP, args=(world_size, args, vae_model, vae, record_function, tp, use_vae, num_codes, return_dict, mean, std, noise_scale), nprocs=world_size, join=True)
            # except Exception as e:
            #     print(f"Error during spawn {e}")

            ## single process
            world_size = 1
            inference_DDP(0, world_size, args, vae_model, vae, record_function, tp, use_vae, num_codes, return_dict, mean=mean, std=std, noise_scale=noise_scale)
            
            real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs = extract_results(return_dict, world_size)
            return real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs
        
        def cal_std(zs):
            dims_to_reduce = [i for i in range(zs.dim()) if i != 1]
            total_std = zs.std().item()
            _mean = zs.mean(dim=dims_to_reduce)
            _std = zs.std(dim=dims_to_reduce)
            return total_std, _mean, _std
        
        real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs = inference()
        if args.noise_scale > 0:
            total_std, _mean, _std = cal_std(zs)
            real_embeddings, fake_embeddings, all_real_videos, all_fake_videos, zs = inference(mean=_mean, std=_std, noise_scale=args.noise_scale)
        
        if args.save_samples:
            torch.save(zs.cpu(), args.save_samples)

        if args.cal_norm:
            total_std, _mean, _std = cal_std(zs)
            print(f"{total_std = } {_mean = } {_std = }")
            if args.save_prediction:
                fname = os.path.join(args.save_dir, args.dataset_list[0], "gt_recon", "mean_std.pth")
                torch.save({'_mean': _mean, '_std': _std}, fname)

        result_str = video_eval(real_embeddings, fake_embeddings, all_real_videos, all_fake_videos)
    else:
        world_size = 1 if args.debug else torch.cuda.device_count()
        manager = torch.multiprocessing.Manager()
        return_dict = manager.dict()

        if args.debug:
            inference_eval(0, world_size, args, vae_model, vae, record_function, use_vae, num_codes, return_dict)
        else:
            spawn(inference_eval, args=(world_size, args, vae_model, vae, record_function, use_vae, num_codes, return_dict), nprocs=world_size, join=True)

        pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, total_usage_bit, total_num_token, all_bit_indices_cat = [], [], 0, 0, 0, 0, 0, 0, 0, 0, []
        for rank in range(world_size):
            pred_xs.append(return_dict[rank]['pred_xs'])
            pred_recs.append(return_dict[rank]['pred_recs'])
            lpips_alex += return_dict[rank]['lpips_alex']
            lpips_vgg += return_dict[rank]['lpips_vgg']
            ssim_value += return_dict[rank]['ssim_value']
            psnr_value += return_dict[rank]['psnr_value']
            num_iter += return_dict[rank]['num_iter']
            total_usage += return_dict[rank]['total_usage']
        pred_xs = np.concatenate(pred_xs, 0)
        pred_recs = np.concatenate(pred_recs, 0)

        result_str = image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, num_codes, total_usage_bit, total_num_token)
        # result_str = inference_eval(args, vae_model, vae, record_function, use_vae, num_codes) 

    print(f"noise scale = {args.noise_scale}")
    print(result_str)
    # save result_str to exp_dir
    basename = os.path.basename(args.vqgan_ckpt)
    match = re.search(r'model_step_(\d+)\.ckpt', basename)
    iter_num = match.group(1) if match else None
    data_prefix_manager.set_data_root(args.data_root, username=args.username)
    ckpt_dir = os.path.dirname(data_prefix_manager(args.vqgan_ckpt))
    use_frames = args.use_frames if args.use_frames else args.sequence_length
    save_dir = os.path.join(ckpt_dir, "evaluation", args.dataset_list[0], f"{args.resolution[0][0]}_{args.resolution[0][1]}", f"{use_frames}")
    os.makedirs(save_dir, exist_ok=True)
    ema_suffix = "_ema" if args.ema == "yes" else ""
    result_name = os.path.join(save_dir, f"result_{iter_num}{ema_suffix}.txt")
    if (not args.save_prediction) and (args.noise_scale == 0):
        with open(result_name, "w") as f:
            f.write(result_str)
    # print('Usage = %.2f'%((total_usage > 0.).sum() / num_codes))
    if args.intermediate_tensor:
        os.system(f"rm -rf {args.intermediate_tensor_dir}")

def add_noise(z, mean, std, noise_scale):
    if noise_scale > 0:
        mean = mean.view(1, mean.shape[0], 1, 1, 1).to(z.device)
        std = std.view(1, std.shape[0], 1, 1, 1).to(z.device)
        z = (z - mean) / std
        noise = torch.randn(z.size()).to(z.device)
        z = (z + noise * noise_scale) * std + mean
    return z

def inference_DDP(rank, world_size, args, vae_model, vae, record_function, tp, use_vae, num_codes, return_dict, mean=None, std=None, noise_scale=0):
    setup(rank, world_size)
    # init data_prefix_manager
    data_prefix_manager.set_data_root(args.data_root, username=args.username)

    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.eval()
    vae = vae.to(f"cuda:{rank}")
    # vae = torch.compile(vae)

    save_dir = os.path.join(args.save_dir, args.dataset_list[0])
    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data = VideoData(args)
    loader = data.val_dataloader()

    i3d = load_fvd_model(f"cuda:{rank}")

    os.makedirs(os.path.join(save_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "recons"), exist_ok=True)

    zs = []
    real_embeddings = []
    fake_embeddings = []

    all_real_videos = []
    all_fake_videos = []

    num_videos = len(loader)
    loader_iter = iter(loader)
    progress_bar = tqdm(total=num_videos, desc=f"Testing {num_videos} batches")
    for batch_idx in range(num_videos):
        if args.turn_on_profiler and tp:
            tp.step()
        batch = next(loader_iter)
        with torch.no_grad():
            input_ = batch['video'] # B C T H W        
            B = input_.shape[0]
            if args.tokenizer in ["hbq_tokenizer"]:
                input_ = input_.to(f"cuda:{rank}").to(torch.bfloat16)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    x_raw, x_recons, z = vae(input_, 0, is_train=False)
                batch['video'] = x_raw.to('cpu').to(torch.float32)
                x_recons = x_recons.to(torch.float32)
            else:
                raise NotImplementedError
                
            if args.tokenizer in ["icvivit", "sd"]:
                x_recons = rearrange(x_recons, "(b t) c h w -> b c t h w", b=B)

            real_videos = torch.clamp(batch['video'] / 2 + 0.5, 0, 1)
            if args.junke_old:
                fake_videos = torch.clamp(x_recons.detach().cpu() + 0.5, 0, 1)
            else:
                fake_videos = torch.clamp(x_recons.detach().cpu() / 2 + 0.5, 0, 1)
            
            use_frames = args.use_frames if args.use_frames else args.sequence_length
            if args.intermediate_tensor:
                folder_name = os.path.join(args.intermediate_tensor_dir, f"{rank}_{batch_idx}")
                os.makedirs(folder_name, exist_ok=True)
                real_file = os.path.join(folder_name, "real_videos.pt")
                fake_file = os.path.join(folder_name, "fake_videos.pt")
                real_videos = real_videos[:,:,:use_frames,...]
                fake_videos = fake_videos[:,:,:use_frames,...]
                torch.save(real_videos.permute(0, 2, 1, 3, 4).squeeze(0), real_file)
                torch.save(fake_videos.permute(0, 2, 1, 3, 4).squeeze(0), fake_file)
                all_real_videos.append(real_file)
                all_fake_videos.append(fake_file)
            else:
                real_videos = real_videos[:,:,:use_frames,...]
                fake_videos = fake_videos[:,:,:use_frames,...]
                all_real_videos.append(real_videos.clone())
                all_fake_videos.append(fake_videos.clone())
            if args.cal_norm or args.save_samples or args.noise_scale > 0:
                zs.append(z)
            real_embedding = get_fvd_logits(shift_dim(real_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=f"cuda:{rank}").cpu()
            real_embeddings.append(real_embedding)
            fake_embedding = get_fvd_logits(shift_dim(fake_videos * 255, 1, -1).byte().data.numpy(), i3d=i3d, device=f"cuda:{rank}").cpu()
            fake_embeddings.append(fake_embedding)

        if args.tokenizer in ['cvivit', "icvivit"] and not use_vae:
            batch_codebook_usage = vq_output["batch_usage"]
            total_usage += batch_codebook_usage

        if args.save_prediction:
            video = torch.cat([real_videos[:,:,:fake_videos.shape[2],:,:], fake_videos], dim=-1)
            b, c, t, h, w = video.shape
            video = video.permute(0, 2, 3, 4, 1).contiguous()
            video = (video.squeeze().detach().cpu().numpy() * 255).astype('uint8')
            os.makedirs(os.path.join(save_dir, "gt_recon"), exist_ok=True)
            this_filename = batch["path"][0].split('/')[-1]
            fname = os.path.join(save_dir, "gt_recon", this_filename)
            import imageio
            imageio.mimsave(fname, video, fps=15)
        
        if args.save_z:
            os.makedirs(os.path.join(save_dir, "gt_recon"), exist_ok=True)
            this_filename = batch["path"][0].split('/')[-1].split(".")[0]
            fname = os.path.join(save_dir, "gt_recon", this_filename+".pt")
            torch.save(z, fname)
        
        if args.save_frames:
            
            def convert_to_uint8(image):
                return (image.detach().cpu().numpy() * 255).astype(np.uint8)

            # artifact_grid_size = 32
            assert real_videos.shape == fake_videos.shape, f"shape of gt and predicted videos are not equal"
            assert real_videos.shape[0] == fake_videos.shape[0] == 1, f"batch size must be 1, real_videos {real_videos.shape[0]}, fake_videos {fake_videos.shape[0]}"
            _real_videos = real_videos.squeeze(0)
            _fake_videos = fake_videos.squeeze(0)
            # h, w = real_videos.shape[-2:]
            # assert (h % artifact_grid_size == 0) and (w % artifact_grid_size == 0), f"height and width of video must be divisible by {artifact_grid_size}"
            
            frame_num = _real_videos.shape[1]
            for frame_idx in range(frame_num):
                real_image = _real_videos[:,frame_idx,:,:]
                fake_image = _fake_videos[:,frame_idx,:,:]
            #     most_different_top_left, max_difference = find_most_different_patch(real_image, fake_image, artifact_grid_size)
                real_image_uint8 = convert_to_uint8(real_image)
                predicted_image_uint8 = convert_to_uint8(fake_image)

                real_image_bgr = cv2.cvtColor(real_image_uint8.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                predicted_image_bgr = cv2.cvtColor(predicted_image_uint8.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

                concatenated_image = np.concatenate((real_image_bgr, predicted_image_bgr), axis=1)
                fname = os.path.join(save_dir, "gt_recon", f"{this_filename}_{frame_idx}.png")
                cv2.imwrite(fname, concatenated_image)
        progress_bar.update(1)

    real_embeddings = torch.cat(real_embeddings, 0)
    fake_embeddings = torch.cat(fake_embeddings, 0)
    zs = torch.cat(zs, 0) if len(zs) > 0 else torch.tensor([])
    if args.intermediate_tensor:
        temp_dict = {
            'real_embeddings':real_embeddings.cpu(),
            'fake_embeddings':fake_embeddings.cpu(),
            'all_real_videos':all_real_videos,
            'all_fake_videos':all_fake_videos,
            'zs': zs.cpu(),
        }
    else:
        all_real_videos = torch.cat(all_real_videos, 0).permute(0, 2, 1, 3, 4)
        all_fake_videos = torch.cat(all_fake_videos, 0).permute(0, 2, 1, 3, 4)
        temp_dict = {
            'real_embeddings':real_embeddings.cpu(),
            'fake_embeddings':fake_embeddings.cpu(),
            'all_real_videos':all_real_videos.cpu(),
            'all_fake_videos':all_fake_videos.cpu(),
            'zs': zs.cpu(),
        }
    # if dist.is_initialized():
    #     dist.barrier()
    return_dict[rank] = temp_dict
    cleanup()

            
def video_eval(real_embeddings, fake_embeddings, all_real_videos, all_fake_videos):
    fake_embeddings = fake_embeddings.to(torch.float64)
    real_embeddings = real_embeddings.to(torch.float64)
    FVD = frechet_distance(fake_embeddings, real_embeddings)
    print(f"FVD: {FVD}")  # can't wait to see this number :)
    del real_embeddings, fake_embeddings

    lpips = calculate_lpips(all_real_videos, all_fake_videos, device="cuda")["value"].values()
    psnr = calculate_psnr(all_real_videos, all_fake_videos)["value"].values()
    ssim = calculate_ssim(all_real_videos, all_fake_videos)["value"].values()
    lpips = np.mean(np.stack(list(lpips)))
    ssim = np.mean(np.stack(list(ssim)))
    psnr = np.mean(np.stack(list(psnr)))

    result_str = f"""
    FVD = {FVD:.4f}
    LPIPS = {lpips:.4f}
    SSIM = {ssim:.4f}
    PSNR = {psnr:.3f}
    """
    return result_str

def inference_eval(rank, world_size, args, vae_model, vae, record_function, use_vae, num_codes, return_dict):
    # Don't remove this setup!!! dist.init_process_group is important for building loader (data.distributed.DistributedSampler)
    setup(rank, world_size) 
    # init data_prefix_manager
    data_prefix_manager.set_data_root(args.data_root, username=args.username)

    device = torch.device(f"cuda:{rank}")

    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device).eval()

    save_dir = os.path.join(args.save_dir, args.dataset_list[0])
    print('generating and saving video to %s...'%save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data = VideoData(args)

    loader = data.val_dataloader()

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    loader_iter = iter(loader)

    pred_xs = []
    pred_recs = []
    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0
    
    num_images = len(loader)
    print(f"Testing {num_images} files")
    num_iter = 0

    total_usage = 0.0
    total_usage_bit = 0.0
    total_num_token = 0
    for batch_idx in tqdm(range(num_images)):
        batch = next(loader_iter)
            
        with torch.no_grad():
            x = batch['video']
            if args.tokenizer in ["hbq_tokenizer"]:
                x_raw, x_recons, z = vae(x.to(device), 0, is_train=False)
                x_recons = x_recons.squeeze(-3).cpu()
            else:
                raise NotImplementedError

            if args.image_recon4video:
                # convert back to image format
                x = x.squeeze(2)
                x_recons = x_recons.squeeze(2)
        
        if args.tokenizer in ["cvivit", "icvivit"] and not use_vae:
        
            # encoding_indices = vq_output["encodings"].detach().cpu()
            code_counts = calculate_batch_codebook_usage_percentage(vq_output["encodings"], num_codes)
            total_counts += code_counts

            batch_codebook_usage = vq_output["batch_usage"]
            total_usage += batch_codebook_usage
        
        paths = batch["path"]
        assert len(paths) == x.shape[0]

        for p, input_ori, recon_ori in zip(paths, x, x_recons):
            if os.path.isabs(p):
                p = "/".join(p.split("/")[6:])
            assert not os.path.isabs(p), f"{p} should not be abspath"
            path = os.path.join(save_dir, "input_recon", os.path.basename(p))
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            
            input_ori = input_ori.unsqueeze(0).to(device)
            input_ = (input_ori + 1) / 2 # [0, 1]
            
            pred_x = inception_model(input_)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()

            recon_ori = recon_ori.unsqueeze(0).to(device)
            recon_ = (recon_ori + 1) / 2 # [0, 1]
            # recon_ = recon_.permute(1, 2, 0).detach().cpu()
            with torch.no_grad():
                pred_rec = inception_model(recon_)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()
            if args.save_prediction:
                if input_.dim() == 4:
                    input_image = input_.squeeze(0)
                if recon_.dim() == 4:
                    recon_image = recon_.squeeze(0)
                input_recon = torch.cat([input_image, recon_image], dim=-1)
                input_recon = Image.fromarray((torch.clamp(input_recon.permute(1, 2, 0).detach().cpu(), 0, 1).numpy() * 255).astype(np.uint8))
                input_recon.save(path)

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            # calculate lpips
            with torch.no_grad():
                lpips_alex += loss_fn_alex(input_ori, recon_ori).sum() # [-1, 1]
                lpips_vgg += loss_fn_vgg(input_ori, recon_ori).sum() # [-1, 1]

            #calculate PSNR and SSIM
            rgb_restored = (recon_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (input_ * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                with torch.no_grad():
                    ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                    psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1
        
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)
    temp_dict = {
        'pred_xs':pred_xs,
        'pred_recs':pred_recs,
        'lpips_alex':lpips_alex.cpu(),
        'lpips_vgg':lpips_vgg.cpu(),
        'ssim_value': ssim_value,
        'psnr_value': psnr_value,
        'num_iter': num_iter,
        'total_usage': total_usage,
        'total_usage_bit': total_usage_bit,
        'total_num_token': total_num_token,
    }
    return_dict[rank] = temp_dict

    # if dist.is_initialized():
    #     dist.barrier()
    cleanup()

def image_eval(pred_xs, pred_recs, lpips_alex, lpips_vgg, ssim_value, psnr_value, num_iter, total_usage, num_codes, total_usage_bit, total_num_token):
    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)
    
    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_iter
    lpips_vgg_value = lpips_vgg / num_iter
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    result_str = f"""
    FID = {fid_value:.4f}
    LPIPS_VGG: {lpips_vgg_value.item():.4f}
    LPIPS_ALEX: {lpips_alex_value.item():.4f}
    SSIM: {ssim_value:.4f}
    PSNR: {psnr_value:.3f}
    """
    return result_str
if __name__ == '__main__':
    main()