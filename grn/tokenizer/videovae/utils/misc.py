# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torch.distributed as dist
import imageio
import os
import random

import math
import numpy as np
from einops import rearrange
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import sys
import pdb as pdb_original
from contextlib import contextmanager

COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"
ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}

def rank_zero_only(fn):
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapped_fn

@rank_zero_only
def print_gpu_usage(model_name) -> None:
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"after {model_name} backward Allocated Memory: {allocated_memory}, Reserved Memory: {reserved_memory}")
    torch.cuda.empty_cache()

def seed_everything(seed=0, allow_tf32=True, benchmark=True, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark  # default False in torch 2.3.1

    # See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # See https://pytorch.org/docs/stable/notes/randomness.html
    torch.use_deterministic_algorithms(deterministic)

    torch.backends.cudnn.allow_tf32 = allow_tf32  # default True in torch 2.3.1
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # default True in torch 2.3.1

# Function to print model summary in table format
@rank_zero_only
def print_model_summary(models):
        # Table headers
        print(f"{'Layer Name':<20} {'Param #':<20}")
        print("="*40)
        total_params = 0
        learnable_params = 0
        for model in models:
            for name, module in model.named_children():
                params = sum(p.numel() for p in module.parameters())
                learnable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_params += params
                params_str = f"{params/1e6:.2f}M"
                print(f"{name:<20} {params_str:<20}")
        print("="*40)
        print(f"Total number of parameters: {total_params/1e6:.2f}M\nLearnable_params: {learnable_params/1e6:.2f}M")

def version_checker(base_version, high_version):
    try:
        from bytedance.ndtimeline import __version__
        from packaging.version import Version
        if Version(__version__) < Version(base_version) or Version(__version__) >= Version(high_version):
            raise RuntimeError(f"bytedance.ndtimeline's version should be >={base_version} <{high_version}, but {__version__} found")
    except ImportError:
        raise RuntimeError(f"bytedance.ndtimeline's version should be >={base_version} <{high_version}")

def is_torch_optim_sch(obj):
    return isinstance(obj, (optim.Optimizer, optim.lr_scheduler.LambdaLR))

def rearranged_forward(x, func):
    if x.ndim == 4:
        x = rearrange(x, "B C H W -> B H W C")
    elif x.ndim == 5:
        x = rearrange(x, "B C T H W -> B T H W C")
    x = func(x)
    if x.ndim == 4:
        x = rearrange(x, "B H W C -> B C H W")
    elif x.ndim == 5:
        x = rearrange(x, "B T H W C -> B C T H W")
    return x

def is_dtype_16(data):
    return data.dtype == torch.float16 or data.dtype == torch.bfloat16

@contextmanager
def set_tf32_flags(flag):
    old_matmul_flag = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_flag = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cudnn.allow_tf32 = flag
    try:
        yield
    finally:
        # Restore the original flags
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_flag
        torch.backends.cudnn.allow_tf32 = old_cudnn_flag

class DataPrefixManager:
    @classmethod
    def set_data_root(cls, data_root, username=""):
        cls._current_data_root = data_root
        cls._username = username

    @classmethod
    def get_work_dir(cls, use_username=True):
        return os.path.join(cls._current_data_root, cls._username)
    
    @classmethod
    def __call__(cls, rel_path, use_username=True, prefix=""):
        return os.path.join(cls.get_work_dir(use_username=use_username), prefix, rel_path)

data_prefix_manager = DataPrefixManager()

def get_last_ckpt(root_dir):
    if not os.path.exists(root_dir): return None
    ckpt_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt') and ('slim_' not in dirpath):
                num_iter = int(filename.split('.ckpt')[0].split('_')[-1])
                ckpt_files[num_iter]=os.path.join(dirpath, filename)
    iter_list = list(ckpt_files.keys())
    if len(iter_list) == 0: return None
    max_iter = max(iter_list)
    return ckpt_files[max_iter]


# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def save_video_grid(video, fname, nrow=None, fps=16):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1).contiguous()

    video = (video.detach().cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    # print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    # print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""
