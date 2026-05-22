from json import load
import os
import argparse
import math
import glob
import time
import logging
from distutils.util import strtobool
from copy import deepcopy
import gc
gc.disable()
import os.path as osp

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.profiler import record_function as torch_record_function
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from videovae.utils.misc import data_prefix_manager, COLOR_BLUE, COLOR_RESET, is_torch_optim_sch
from videovae.utils.distributed import init_distributed_mode, reduce_losses, average_losses, _FSDP
from videovae.utils.ema import update_ema, requires_grad

from videovae.models.discriminator import ImageDiscriminator, VideoDiscriminator
from videovae.data import VideoData
from videovae.modules import build_lpips_model
from videovae.modules.loss import get_disc_loss, adopt_weight
from videovae.utils.misc import get_last_ckpt, seed_everything, print_gpu_usage, print_model_summary, version_checker
from videovae.utils.init_models import init_vae_only, init_vit_from_image, resume_from_ckpt, init_cnn_from_image, load_cnn
from videovae.utils.nan_detector import NanDetector
from videovae.utils.arguments import MainArgs, add_model_specific_args, init_args, format_args
from videovae.utils.scheduler import get_lambda
from videovae.utils.mfu import register_mfu_hook, get_mfu, get_tflops, get_tflops_dict

from videovae.utils.context_parallel import ContextParallelUtils as cp

def save_model(fsdp_model, rank, model_path, global_step):
    # FSDP推荐用 state_dict_type=FULL_STATE_DICT 来保存
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state_dict = fsdp_model.state_dict()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'vae': state_dict, 'step': global_step}, model_path)
        print(f"模型已保存到 {model_path}")

# enable_timeline_sdk = strtobool(os.getenv("GenAI_USE_TIMELINE_SDK", "0"))
enable_timeline_sdk = False

def split_to_ranks(x):
    bs = x.shape[0]
    cp_size = cp.get_cp_size()
    if cp_size > 1 and bs % cp_size == 0:
        cp_rank = cp.get_cp_rank()
        return x.chunk(cp_size, dim=0)[cp_rank]
    else:
        return x

if enable_timeline_sdk:
    try:
        import bytedance.ndtimeline as ndtimeline
    except ImportError:
        print(f"import vescale.ndtimeline failed, skipped")
        enable_timeline_sdk = False

def init_data_scheduler(video_ranks_ratio: float = -1.0, cp_size: int = 1):
    if video_ranks_ratio < 0:
        return None,None

    cp_size = max(1, cp_size)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    video_ranks = list(range(int((world_size * video_ranks_ratio) // cp_size) * cp_size)) # align to cp_size for video
    image_ranks = list(range(len(video_ranks), world_size))

    print(f"[info] video_ranks: {video_ranks}, image_ranks: {image_ranks}")

    if rank in image_ranks:
        group = torch.distributed.new_group(image_ranks)
        dataset_type_on_this_rank = "image"
    else:
        group = torch.distributed.new_group(video_ranks)
        dataset_type_on_this_rank = "video"

    return group, dataset_type_on_this_rank

def main():
    parser = argparse.ArgumentParser()
    parser = MainArgs.add_main_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    args, unknown = parser.parse_known_args()
    args, parser, vae_model = add_model_specific_args(args, parser)
    args = parser.parse_args()

    args = init_args(args) # post process args

    # init data_prefix_manager
    data_prefix_manager.set_data_root(args.data_root, username=args.username)
    args.default_root_dir = data_prefix_manager(args.default_root_dir)

    # Setup DDP:
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed_everything(args.seed)
    torch.cuda.set_device(device)

    # init context parallel
    cp_cfg = {"cp_size": args.context_parallel_size}
    cp.initialize_context_parallel(cp_cfg)

    ds_group, ds_type = init_data_scheduler(args.video_ranks_ratio, cp_size = args.context_parallel_size)

    # Setup an experiment folder:
    checkpoint_dir = f"{args.default_root_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        script_str = format_args(args)
        with open(os.path.join(args.default_root_dir, "script.sh"), "w") as f:
            f.write(script_str)
        print(f"{COLOR_BLUE}Experiment directory created at {args.default_root_dir}{COLOR_RESET}")

        import wandb
        wandb_project = "HBQ_Tokenizer"
        wandb.init(
            project=wandb_project,
            name=os.path.basename(os.path.normpath(args.default_root_dir)),
            dir=args.default_root_dir,
            config=args,
            mode="offline" if args.debug else "online"
        )
    
    # init model    
    vae = vae_model(args).to(device)
    if rank == 0:
        model_arch_save_path = os.path.join(args.default_root_dir, "model_arch.txt")
        print(f"{COLOR_BLUE}Logging model architecture at {model_arch_save_path}{COLOR_RESET}")
        with open(model_arch_save_path, "w") as f:
            f.write(str(vae))
    image_disc = ImageDiscriminator(args).to(device)
    video_disc = VideoDiscriminator(args).to(device)

    # init optimizers and schedulers
    if args.optim_type == "Adam":
        vae_optim = torch.optim.Adam
    elif args.optim_type == "AdamW":
        vae_optim = torch.optim.AdamW
    if args.disc_optim_type is None:
        disc_optim = vae_optim
    elif args.disc_optim_type == "rmsprop":
        disc_optim = torch.optim.RMSprop

    def get_param_groups(model):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if len(param.shape) == 1 or name.endswith(".bias") or ('scale_learnable_parameters' in name):
                    no_decay.append(param)
                    print(f'disable weight deacy for {name}')
                else:
                    decay.append(param)
        optimizer_grouped_parameters = [
            {'params': decay, 'weight_decay': 0.01},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    opt_vae = vae_optim(get_param_groups(vae), lr=args.lr, betas=(args.beta1, args.beta2))
    if disc_optim == torch.optim.RMSprop:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier)
        opt_video_disc = disc_optim(video_disc.parameters(), lr=args.lr * args.dis_lr_multiplier)
    else:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier, betas=(args.beta1, args.beta2))
        opt_video_disc = disc_optim(video_disc.parameters(), lr=args.lr * args.dis_lr_multiplier, betas=(args.beta1, args.beta2))

    if args.scheduler == "no":
        sch_vae, sch_image_disc, sch_video_disc = None, None, None
    else:
        lr_lambda = get_lambda(args)
        sch_vae = optim.lr_scheduler.LambdaLR(opt_vae, lr_lambda)
        sch_image_disc = optim.lr_scheduler.LambdaLR(opt_image_disc, lr_lambda)
        sch_video_disc = optim.lr_scheduler.LambdaLR(opt_video_disc, lr_lambda)

    ### ema
    ema = None
    if args.ema == "yes":
        ema = deepcopy(vae).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        print(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
        update_ema(ema, vae, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode

    model_optims = {
        "vae" : vae,
        "image_disc" : image_disc,
        "video_disc" : video_disc,
        "opt_vae" : opt_vae,
        "opt_image_disc" : opt_image_disc,
        "opt_video_disc" : opt_video_disc,
        "sch_vae" : sch_vae,
        "sch_image_disc" : sch_image_disc,
        "sch_video_disc" : sch_video_disc,
        "ema": ema, 
    }

    ### Resume from checkpoint in default_root_dir or load pretrained weights if specified
    ckpt_path = None
    assert not args.default_root_dir is None # required argument
    ckpt_path = get_last_ckpt(args.default_root_dir)
    init_step = 0
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model_optims, init_step = resume_from_ckpt(state_dict, model_optims, load_optims=args.zero<=0, remove_disc=args.remove_disc, ckpt_path=ckpt_path, args=args)
    elif args.pretrained is not None:
        args.pretrained = data_prefix_manager(args.pretrained)
        # read weight
        state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        
        if args.pretrained_ema == "yes":
            state_dict["vae"] = state_dict["ema"] # replace vae weights with ema weight
        # load model
        if args.pretrained_mode == "weights":
            model_optims, _ = resume_from_ckpt(state_dict, model_optims, load_optims=False, remove_disc=args.remove_disc, remove_enlarge_factors=args.remove_enlarge_factors, args=args) # load all models and optims
            del state_dict
        else:
            raise NotImplementedError
        print(f"Successfully loaded ckpt {args.pretrained}, pretrained_mode {args.pretrained_mode}")

    # init dataloader
    data = VideoData(args, ds_group = ds_group, ds_type = ds_type)
    dataloaders = data.train_dataloader()
    dataloader_iters = [iter(loader) for loader in dataloaders]
    ### init epoch in resuming
    dataloader_init_epoch = (
        init_step if init_step > 0 # in case of resuming
        else args.dataloader_init_epoch if args.dataloader_init_epoch > 0 # in case of fintuning
        else 0
    )
    data_epochs = [dataloader_init_epoch for _ in dataloaders]
    for idx in range(len(dataloaders)):
        print(f"Reset the {idx}th dataloader as epoch {data_epochs[idx]}")
        if hasattr(dataloaders[idx], "sampler"):
            dataloaders[idx].sampler.set_epoch(data_epochs[idx])
        else:
            raise NotImplementedError

    ### torch.compile after loading all weights
    print_model_summary([vae, image_disc, video_disc])
    if args.zero > 0:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
        )
        def my_policy(
            module: torch.nn.Module,
            recurse: bool,
            **kwargs,
        ) -> bool:
            return True
        auto_wrap_policy = my_policy
        vae = FSDP(
            vae, 
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=None,
        ).to(device)
        # vae = _FSDP(vae, device, args.zero)
    else:
        vae = DDP(vae.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb, find_unused_parameters=True)

    image_disc = DDP(image_disc.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    video_disc = DDP(video_disc.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)

    image_perceptual_model, video_perceptual_model = build_lpips_model(args)
    image_perceptual_model = image_perceptual_model.to(device)
    video_perceptual_model = video_perceptual_model.to(device)

    if args.compile == "yes":
        if args.vf_weight > 0 and args.vf_weight_approx < 0:
            torch._functorch.config.donated_buffer = False # This backward function was compiled with non-empty donated buffers which requires create_graph=False and retain_graph=False
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 4096
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.optimize_ddp = False if args.use_checkpoint else True

        for k in model_optims:
            if k != "ema" and model_optims[k] and not is_torch_optim_sch(model_optims[k]):
                print(f"compiling model {k}")
                if k == "vae":
                    model_optims[k].encoder.compile()#options={'fx_graph_cache':True})
                    model_optims[k].decoder.compile()#options={'fx_graph_cache':True})
                else:
                    model_optims[k].compile()#options={'fx_graph_cache':True})
        print(f"Successfully compiled all models")

    disc_loss = get_disc_loss(args.disc_loss_type)
    
    if enable_timeline_sdk:
        version_checker("2.0.0", "3.0.0")
        bmq_cluster = os.getenv('CUDA_TIMER_STREAM_KAFKA_CLUSTER', 'bmq_bigbang_3rd')
        bmq_topic = os.getenv('CUDA_TIMER_STREAM_KAFKA_TOPIC', 'megatron_cuda_timer_tracing_original')
        ndtimeline.init_ndtimers(
            mode="fsdp",
            mesh_shape=(world_size,),
            world_size=world_size,
            enable_streamer=True,
            post_handlers=[ndtimeline.handlers.MQNDHandler(mq_sinks=[ndtimeline.handlers.format_mq_sink(bmq_cluster, bmq_topic)])],
        )
        ndtimeline.set_global_step(init_step)
        print(f"init timeline successfully")

    # init profiler
    def trace_handler(p):
        p.export_chrome_trace(os.path.join(args.default_root_dir, f"trace_step_{p.step_num}_rank_{rank}.json.gz"))

    if args.turn_on_profiler:
        print(f"start to init profiler")
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
        print(f"finish to init profiler")
    else:
        record_function = nullcontext

    start_time = time.time()
    cnt = 0
    debug_root_dir = data_prefix_manager(f"debug/{cnt}")
    while os.path.exists(debug_root_dir):
        cnt += 1
        debug_root_dir = data_prefix_manager(f"debug/{cnt}")
    os.makedirs(debug_root_dir, exist_ok=True)
    if cp.get_cp_rank() == 0:
        cp_group = cp.get_cp_group_rank()
        debug_f = open(os.path.join(debug_root_dir, f"rank_{rank}_cp_group_{cp_group}.txt"), "w")
    else:
        debug_f = None
    for global_step in range(init_step, args.max_steps):
        # if args.turn_on_profiler and tp:
        #     tp.step()
        loss_dicts = []
        
        if global_step == args.discriminator_iter_start - args.disc_pretrain_iter:
            logging.info(f"discriminator begins pretraining ")
        if global_step == args.discriminator_iter_start:
            log_str = "add GAN loss into training"
            if args.disc_pretrain_iter > 0:
                log_str += ", discriminator ends pretraining"
            logging.info(log_str)
        
        for idx in range(len(dataloader_iters)):
            try:
                _batch = next(dataloader_iters[idx])
            except StopIteration:
                data_epochs[idx] += 1
                print(f"Reset the {idx}th dataloader as epoch {data_epochs[idx]}")
                dataloaders[idx].sampler.set_epoch(data_epochs[idx])
                dataloader_iters[idx] = iter(dataloaders[idx]) # update dataloader iter
                _batch = next(dataloader_iters[idx])
            except Exception as e:
                raise e 
            x = _batch["video"]

            _type = _batch["type"][0]

            if _type == "image" and ds_type is None:
                x = split_to_ranks(x)

            disc_factor = 1.
            with NanDetector(vae) if args.enable_nan_detector else nullcontext():
                with record_function("vae"):
                    if _type == "image":
                        x, x_recon, flat_frames, flat_frames_recon, vae_loss_dict, vae_log_dict = vae(x, disc_factor, image_disc=image_disc, image_perceptual_model=image_perceptual_model)
                    elif _type == "video":
                        if debug_f is not None:
                            debug_f.write(f'step {idx}, {_batch["path"]}\n')
                        x, x_recon, flat_frames, flat_frames_recon, vae_loss_dict, vae_log_dict = vae(
                            x, disc_factor, 
                            image_disc=image_disc, video_disc=video_disc, 
                            image_perceptual_model=image_perceptual_model, 
                            video_perceptual_model=video_perceptual_model,
                        )
                    g_loss = sum(vae_loss_dict.values())
                    opt_vae.zero_grad()
                    g_loss.backward()
                    # print_gpu_usage("vae")
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                    
                    # from pnp.utils import detect_anomalous_params
                    # detect_anomalous_params(g_loss, vae)

                    opt_vae.step()
                    opt_vae.zero_grad() # free memory
                    if args.ema == "yes":
                        update_ema(ema, vae.module)
                        
                with record_function("disc"):
                    disc_loss_dict = {}
                    # args.discriminator_iter_start=-1 args.disc_pretrain_iter=0
                    discloss = d_image_loss = d_video_loss = torch.tensor(0.).to(x.device)
                    ### enable pool warmup
                    for disc_step in range(args.disc_optim_steps):
                        require_optim = False
                        if _type == "image":
                            if args.image_disc_weight > 0:
                                require_optim = True
                                logits_image_real = image_disc(x, pool_name="real")
                                logits_image_fake = image_disc(x_recon.detach(), pool_name="fake")
                                d_image_loss = disc_loss(logits_image_real, logits_image_fake)
                                discloss = d_image_loss * args.image_disc_weight
                                disc_loss_dict["train/logits_image_real"] = logits_image_real.mean().detach()
                                disc_loss_dict["train/logits_image_fake"] = logits_image_fake.mean().detach()
                                disc_loss_dict["train/d_image_loss"] = discloss.detach()
                            opt_discs, sch_discs = [opt_image_disc], [sch_image_disc]
                        elif _type == "video":
                            if args.image_disc_weight > 0 and args.gan_image4video == "yes":
                                require_optim = True
                                logits_image_real = image_disc(flat_frames.detach(), pool_name="real")
                                logits_image_fake = image_disc(flat_frames_recon.detach(), pool_name="fake")
                                d_image_loss = disc_loss(logits_image_real, logits_image_fake)
                                disc_loss_dict["train/logits_image_real"] = logits_image_real.mean().detach()
                                disc_loss_dict["train/logits_image_fake"] = logits_image_fake.mean().detach()
                                disc_loss_dict["train/d_image_loss"] = (d_image_loss * args.image_disc_weight).detach()
                            if args.video_disc_weight > 0:
                                require_optim = True
                                logits_video_real = video_disc(x.detach(), pool_name="real")
                                logits_video_fake = video_disc(x_recon.detach(), pool_name="fake")
                                d_video_loss = disc_loss(logits_video_real, logits_video_fake)
                                disc_loss_dict["train/logits_video_real"] = logits_video_real.mean().detach()
                                disc_loss_dict["train/logits_video_fake"] = logits_video_fake.mean().detach()
                                disc_loss_dict["train/d_video_loss"] = (d_video_loss * args.video_disc_weight).detach()
                            discloss = d_image_loss * args.image_disc_weight + d_video_loss * args.video_disc_weight
                            opt_discs, sch_discs = [opt_image_disc, opt_video_disc], [sch_image_disc, sch_video_disc]
                        discloss = disc_factor * discloss

                        if require_optim:
                            for opt_disc in opt_discs:
                                opt_disc.zero_grad()
                            discloss.backward()
                            # print_gpu_usage("disc")
                            if args.max_grad_norm_disc > 0:
                                torch.nn.utils.clip_grad_norm_(image_disc.parameters(), args.max_grad_norm_disc)
                                torch.nn.utils.clip_grad_norm_(video_disc.parameters(), args.max_grad_norm_disc)

                            for opt_disc in opt_discs:
                                opt_disc.step()
                            for opt_disc in opt_discs:
                                opt_disc.zero_grad() # free memory

            with record_function("loss"):
                loss_dict = {**vae_loss_dict, **disc_loss_dict, **vae_log_dict}
                if (global_step+1) % args.log_every == 0:
                    reduced_loss_dict = reduce_losses(loss_dict)
                else:
                    reduced_loss_dict = {}
                loss_dicts.append(reduced_loss_dict)
        
        # update scheduler   
        if not sch_vae is None:
            sch_vae.step()
        for sch_disc in sch_discs:
            if not sch_disc is None:
                sch_disc.step()

        # if enable_timeline_sdk:
            # ndtimeline.inc_step()

        if (global_step+1) % args.log_every == 0:
            avg_loss_dict = average_losses(loss_dicts)
            torch.cuda.synchronize()
            end_time = time.time()
            iter_speed = (end_time - start_time) / args.log_every

            if args.mfu_logging == "yes":
                tflops = get_tflops() / args.log_every
                tflops_log_str = f"tflops={tflops:.1f}, "
                tflops_dict = get_tflops_dict(args.log_every)
                tflops_dict_log_str = f"tflops_Dict={tflops_dict}, "
                mfu = get_mfu(iter_speed) / args.log_every
                mfu_log_str = f"mfu={mfu:.3f}, "
            else:
                tflops_log_str = ""
                tflops_dict_log_str = ""
                mfu_log_str = ""

            if rank == 0:
                avg_loss_dict["lr"] = opt_vae.param_groups[0]['lr']
                for key, value in avg_loss_dict.items():
                    wandb.log({key: value}, step=global_step)
                
                recons_loss_sum, video_perceptual_loss_sum = 0., 0.
                for key in avg_loss_dict:
                    if 'recon_loss' in key:
                        recons_loss_sum += avg_loss_dict[key]
                    if 'video_perceptual_loss' in key:
                        video_perceptual_loss_sum += avg_loss_dict[key]
                
                print(f'global_step={global_step}, recon_loss={recons_loss_sum:.4f}, ' \
                            f'video_perceptual_loss={video_perceptual_loss_sum:.4f}, ' \
                            f'iter_speed={iter_speed:.2f}s, ' \
                            f'{mfu_log_str}' \
                            f'{tflops_log_str}' \
                            f'{tflops_dict_log_str}' \
                            )
            start_time = time.time()
            if enable_timeline_sdk:
                ndtimeline.flush()
        
        if (global_step+1) % args.ckpt_every == 0 and global_step != init_step:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{global_step}.ckpt')
            if args.zero > 0:
                save_model(vae, rank, checkpoint_path, global_step)
            else:
                if rank == 0:
                    save_dict = {}
                    for k in model_optims:
                        model = model_optims[k]
                        save_dict[k] = None if model is None \
                            else model.module.state_dict() if hasattr(model, "module") \
                            else model.state_dict()
                    torch.save({
                        'step': global_step,
                        **save_dict,
                    }, checkpoint_path)
                    print(f'Checkpoint saved at step {global_step}')
        
        if (global_step+1) % args.manual_gc_interval == 0:
            gc.collect()
    
if __name__ == '__main__':
    main()
