import os
import math
from datetime import datetime
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.modules.conv import _ConvNd
from torch.utils.checkpoint import TorchDispatchMode


def get_device_tflops():
    peak_tflops = -1
    arch = torch.cuda.get_device_capability()
    if arch[0] == 8 and arch[1] == 0:  # A100/A800
        peak_tflops = 312
    elif arch[0] == 9 and arch[1] == 0:  # H100/H800
        peak_tflops = 989
    else:
        print(f"unknown default tflops of device capability {arch[0]}.{arch[1]}")
    return peak_tflops


class NullCtx(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class DisableMfu(NullCtx):
    def __enter__(self):
        super().__enter__()
        self.old_flop_enable = Flops.enable
        Flops.enable = False

    def __exit__(self, *args, **kwargs):
        Flops.enable = self.old_flop_enable
        super().__exit__(*args, **kwargs)


def context_fn():
    return NullCtx(), DisableMfu()


class CustomFlops(ABC):
    """
    For functions,
    1. run the func within CustomFlops
    2. implement the hook `flops`
    to support register_forward_hook
    """
    @abstractmethod
    def flops(self, args, kwargs, output) -> dict:
        pass


def conv_flops_func(module, args, kwargs, output):
    return 2 * math.prod(module.kernel_size) * module.in_channels * output.numel()


def linear_flops_func(module, args, kwargs, output):
    return 2 * module.in_features * output.numel()


def layernorm_flops_func(module, args, kwargs, output):
    return 4 * output.numel()


def groupnorm_flops_func(module, args, kwargs, output):
    return 2 * output.numel()


def syncbatchnorm_flops_func(module, args, kwargs, output):
    return 2 * output.numel()


basic_flops_func = {
    _ConvNd: conv_flops_func,
    nn.Linear: linear_flops_func,
    nn.LayerNorm: layernorm_flops_func,
    nn.GroupNorm: groupnorm_flops_func,
    nn.SyncBatchNorm: syncbatchnorm_flops_func,
}

@torch._dynamo.disable()
def calculate_flops(module, args, kwargs, output):
    flops = 0
    flops_dict = {}
    if isinstance(module, CustomFlops):
        flops_dict = module.flops(args, kwargs, output)
    else:
        flops_func = basic_flops_func[module._base_m]
        flops_dict = {module.__class__.__name__: flops_func(module, args, kwargs, output)}

    for module_class, module_flops in flops_dict.items():
        if module_class not in Flops.module_flops_dict:
            Flops.module_flops_dict[module_class] = module_flops * (3 if module.training else 1)
        else:
            Flops.module_flops_dict[module_class] += module_flops * (3 if module.training else 1)
    
    flops = sum(list(flops_dict.values()))
    Flops.flops += flops * (3 if module.training else 1)


class Flops:
    handlers = []
    flops = 0
    enable = True
    module_flops_dict = {}

    @staticmethod
    def reset():
        tmp = Flops.flops
        Flops.flops = 0
        Flops.module_flops_dict = {}
        return tmp

    @staticmethod
    def _hook(module, args, kwargs, output):
        if not Flops.enable:
            return

        if module.training and not torch.is_grad_enabled():
            # activation checkpoint mode
            return
        calculate_flops(module, args, kwargs, output)

    @staticmethod
    def _dfs_register_hooks(parent_name: str, cur_m: nn.Module):
        for name, m in cur_m.named_children():
            # custom hooks
            if isinstance(m, CustomFlops):
                assert isinstance(m, nn.Module)
                Flops.handlers.append(
                    m.register_forward_hook(Flops._hook, with_kwargs=True)
                )
                continue
            # built-in hooks
            is_registered = False
            for base_m, flops_func in basic_flops_func.items():
                if isinstance(m, base_m):
                    m._base_m = base_m
                    Flops.handlers.append(
                        m.register_forward_hook(Flops._hook, with_kwargs=True)
                    )
                    is_registered = True
                    break
            if not is_registered:
                Flops._dfs_register_hooks(parent_name + "." + name, m)

    @staticmethod
    def unwrap(self):
        for hdl in Flops.handlers:
            hdl.remove()



def register_mfu_hook(model):
    Flops._dfs_register_hooks("root", model)


def get_tflops():
    return Flops.flops / 1e12


def get_tflops_dict(record_iters=1):
    tflops_dict = {module: round(flops / record_iters/ 1e12, 3) for module, flops in Flops.module_flops_dict.items()}
    return tflops_dict


def get_mfu(iter_time):
    # compute MFU
    ideal_TFLOPS = get_device_tflops()
    achieve_TFLOPs = Flops.reset() / 1e12
    mfu = achieve_TFLOPs / iter_time / ideal_TFLOPS
    return mfu
