# from https://github.com/FoundationVision/LlamaGen/blob/main/utils/distributed.py
import os
import sys
import glob
import torch
import subprocess
import torch.distributed as dist
import datetime
import logging
import builtins

from videovae.utils.misc import rank_zero_only, COLOR_BLUE, COLOR_RESET

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)


def setup_for_distributed(is_master, logging_dir=""):
    builtin_print = builtins.print
    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)
    builtins.print = print
    
    if is_master:
        os.makedirs(logging_dir, exist_ok=True)
        existing_logs = glob.glob(os.path.join(logging_dir, 'log_out_*.txt'))
        log_numbers = [int(log.split('.txt')[0].split('_')[-1]) for log in existing_logs]
        next_log_number = max(log_numbers) + 1 if log_numbers else 1
        
        log_out_path = os.path.join(logging_dir, f'log_out_{next_log_number}.txt')
        log_err_path = os.path.join(logging_dir, f'log_err_{next_log_number}.txt')
        
        print(f"{COLOR_BLUE}stdout will be written to {log_out_path}{COLOR_RESET}")
        print(f"{COLOR_BLUE}stderr will be written to {log_err_path}{COLOR_RESET}")
        
        # sys.stdout = Tee(sys.stdout, open(log_out_path, 'w'))
        # sys.stderr = Tee(sys.stderr, open(log_err_path, 'w'))

class Tee(object):
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def init_distributed_mode(args, timeout_minutes=15):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=timeout_minutes * 60)
                                         )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0, args.default_root_dir)

def _FSDP(model: torch.nn.Module, device, zero) -> FSDP:
    def my_policy(
        module: torch.nn.Module,
        recurse: bool,
        **kwargs,
    ) -> bool:
        return True
    auto_wrap_policy = my_policy
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy={1:ShardingStrategy.HYBRID_SHARD, 2:ShardingStrategy.SHARD_GRAD_OP, 3:ShardingStrategy.FULL_SHARD}.get(zero),
        mixed_precision=MixedPrecision(
            param_dtype=torch.float,
            reduce_dtype=torch.float,
            buffer_dtype=torch.float,
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def reduce_losses(loss_dict, dst=0):
    loss_names = list(loss_dict.keys())
    loss_tensor = torch.stack([loss_dict[name] for name in loss_names])

    dist.reduce(loss_tensor, dst=dst, op=dist.ReduceOp.SUM)
    # Only average the loss values on the destination rank
    if dist.get_rank() == dst:
        loss_tensor /= dist.get_world_size()
        averaged_losses = {name: loss_tensor[i].item() for i, name in enumerate(loss_names)}
    else:
        averaged_losses = {name: None for name in loss_names}
    
    return averaged_losses

@rank_zero_only
def average_losses(loss_dict_list):
    sum_dict = {}
    count_dict = {}
    for loss_dict in loss_dict_list:
        for key, value in loss_dict.items():
            if key in sum_dict:
                sum_dict[key] += value
                count_dict[key] += 1
            else:
                sum_dict[key] = value
                count_dict[key] = 1

    avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return avg_dict
