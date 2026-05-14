import argparse
import datetime
import numpy as np
import os
import time
import functools
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from grn.utils_c2i.crop import center_crop_arr
import grn.utils_c2i.misc as misc

import copy
from grn.utils_c2i.engine import train_one_epoch, evaluate
from grn.utils import wandb_utils as wandb_utils

from grn.utils_c2i.denoiser import Denoiser
from grn.models.grn_c2i import GRNblock


def get_args_parser():
    parser = argparse.ArgumentParser('GRN', add_help=False)

    # architecture
    parser.add_argument('--model', default='GRN_B', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', type=int, default=0, choices=[0,1],
                        help='Whether to evaluate the model online')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--hbq_round', default=4, type=int,)
    parser.add_argument('--vae_latent', default=16, type=int,)
    parser.add_argument('--in_channels', default=3, type=int,)
    parser.add_argument('--method', default='GRN_ind', type=str, choices=['GRN_ind', 'GRN_bit'])
    parser.add_argument('--vae_path', default='', type=str,)
    parser.add_argument('--tau', default=1.0, type=float,)
    parser.add_argument('--wandb', default=1, type=int, choices=[0,1])
    parser.add_argument('--generation_dir', default='/tmp', type=str)
    parser.add_argument('--clip_grad_norm', default=1., type=float)
    parser.add_argument('--use_fsdp_train', default=0, type=int, choices=[0, 1])
    parser.add_argument('--delete_images', default=1, type=int, choices=[0, 1])
    parser.add_argument('--complexity_aware_wp', default=5, type=int)
    parser.add_argument('--complexity_aware_k', default=1, type=float)
    parser.add_argument('--complexity_aware_b', default=0, type=float)
    parser.add_argument('--complexity_aware_tmin', default=20, type=float)
    parser.add_argument('--inner_shard_degree', default=8, type=int)
    parser.add_argument('--patch_size', default=1, type=int)
    parser.add_argument('--convert_type', default='', type=str)
    parser.add_argument('--mask_group_size', default=-1, type=int)
    parser.add_argument('--grn_shift_factor', default=1., type=float)
    parser.add_argument('--use_focal_loss', default=0, type=int, choices=[0, 1])
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
        if args.wandb:
            entity = os.environ["EXP_NAME"]
            project = os.environ["PROJECT"]
            wandb_utils.wandb.init(project=project, name=entity, config={})
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    # ininitalize vae
    from grn.models.hbq_tokenizer import HBQ_Tokenizer
    vae = HBQ_Tokenizer(args=args, latent_channels=args.vae_latent, encoder_out_type='feature_tanh')
    vae.eval()
    vae = vae.to('cuda')
    for param in vae.parameters():
        param.requires_grad = False
    state_dict = torch.load(args.vae_path, map_location='cuda')
    if 'ema' in state_dict:
        print(f'Load ema vae weights')
        state_dict = state_dict['ema']
    else:
        print(f'Load non ema vae weights')
        state_dict = state_dict['vae']
    print('Load vae: ', vae.load_state_dict(state_dict, assign=True))

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    if args.use_fsdp_train:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GRNblock},
        )
        if args.inner_shard_degree > 0:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
            world_size = misc.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree <= world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD
            device_mesh = None
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, 
                reduce_dtype=torch.bfloat16, 
                buffer_dtype=torch.bfloat16
            ),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy, 
            use_orig_params=True,
            device_mesh=device_mesh,
        )
        model_without_ddp = model
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    # checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    checkpoint_path = args.resume if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        if args.use_fsdp_train:
            # For FSDP, load EMA state dict into model temporarily to set ema_params
            model_without_ddp.load_state_dict(checkpoint['model_ema1'])
            model_without_ddp.module.ema_params1 = [p.detach().clone() for p in model_without_ddp.parameters()]
            
            model_without_ddp.load_state_dict(checkpoint['model_ema2'])
            model_without_ddp.module.ema_params2 = [p.detach().clone() for p in model_without_ddp.parameters()]
            
            # Restore model
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            ema_state_dict1 = checkpoint['model_ema1']
            ema_state_dict2 = checkpoint['model_ema2']
            model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        
        print("Resumed checkpoint from", args.resume)

        try:
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                if args.use_fsdp_train:
                    opt_state = FSDP.optim_state_dict_to_load(
                        model_without_ddp, optimizer, checkpoint['optimizer']
                    )
                    optimizer.load_state_dict(opt_state)
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded optimizer & scaler state!")
        except:
            print("Failed to load optimizer & scaler state! Just load checkpoint.")
        args.start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        if args.use_fsdp_train:
            model_without_ddp.module.ema_params1 = [p.detach().clone() for p in model_without_ddp.parameters()]
            model_without_ddp.module.ema_params2 = [p.detach().clone() for p in model_without_ddp.parameters()]
        else:
            model_without_ddp.ema_params1 = [p.detach().clone() for p in model_without_ddp.parameters()]
            model_without_ddp.ema_params2 = [p.detach().clone() for p in model_without_ddp.parameters()]
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer, vae=vae)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args, vae=vae)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            if misc.is_main_process():
                from grn.utils.safe_rm import safe_remove
                safe_remove(f'{args.output_dir}/checkpoint-tmp_*.pth', args.output_dir)
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name=f"tmp_{epoch}"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, vae=vae)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
