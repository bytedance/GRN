import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_norm(norm_type):
    if norm_type == "spatial-group":
        return SpatialGroupNorm
    elif norm_type == "rms":
        return RMS_norm
    elif norm_type == "group":
        return nn.GroupNorm
    else:
        raise NotImplementedError

class RMS_norm(nn.Module):

    def __init__(self, num_channels, channel_first=True, bias=False, **kwargs):
        super().__init__()
        broadcastable_dims = (1, 1, 1)
        shape = (num_channels, *broadcastable_dims)

        self.channel_first = channel_first
        self.scale = num_channels**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class SpatialGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super(SpatialGroupNorm, self).__init__(*args, **kwargs)
    
    def shard_norm(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        with torch.amp.autocast("cuda", torch.float32):
            for _i in range(x.shape[0]):
                x[_i:_i+1,...] = super(SpatialGroupNorm, self).forward(x[_i:_i+1,...])
        x = x.to(dtype=dtype)
        return x

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        assert x.ndim == 5
        T = x.shape[2]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        try:
            x = super(SpatialGroupNorm, self).forward(x)
        except:
            x = self.shard_norm(x) # shard norm if OOM fallback
        x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        x = x.to(dtype=dtype)
        return x

class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type, norm_axis="spatial"):
        super().__init__()
        self.norm_axis = norm_axis
        assert norm_type in ['group', 'batch', "no"]
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0: 
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # Runtime Error: grad inplace if set track_running_stats to True
        elif norm_type == 'no':
            self.norm = nn.Identity()
    
    def _norm(self, x):
        try:
            x = self.norm(x)
        except:
            device = x.device
            self.norm_cpu = self.norm.cpu()
            x = self.norm_cpu(x.cpu().pin_memory()).to(device=device)
        return x

    def shard_norm(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        with torch.amp.autocast("cuda", torch.float32):
            for _i in range(x.shape[0]):
                x[_i:_i+1,...] = self.norm(x[_i:_i+1,...])
        x = x.to(dtype=dtype)
        return x

    def forward(self, x):
        if self.norm_axis == "spatial":
            if type(x) == list:
                for i in range(len(x)):
                    x[i] = self.norm(x[i])
                return x
            if x.ndim == 4:
                try:
                    x = self.norm(x)
                except:
                    x = self.shard_norm(x)
            else:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                # x = self.shard_norm(x)
                try:
                    x = self.norm(x)
                except:
                    x = self.shard_norm(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
        elif self.norm_axis == "spatial-temporal":
            x = self._norm(x)
        else:
            raise NotImplementedError
        return x

def l2norm(t):
    return F.normalize(t, dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# https://github.com/huggingface/transformers/blob/2f12e408225b1ebceb0d2f701ce419d46678dc31/src/transformers/models/llama/modeling_llama.py#L76
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, sp_slice=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if sp_slice is None:
            return (self.weight * hidden_states).to(input_dtype)
        else:
            return (self.weight[sp_slice] * hidden_states).to(input_dtype)  # torch.float32 * torchbfloat16 in DDP will cast to torch.float32
