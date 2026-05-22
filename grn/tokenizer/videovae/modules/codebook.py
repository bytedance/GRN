# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from enum import unique
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from videovae.utils.misc import shift_dim

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0, usage_sigma=0.99, fp32_quant=False):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
    

    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        # Flatten the batch of encoding indices into a single 1D tensor
        all_indices = batch_encoding_indices.flatten()
        
        # Obtain the total number of encoding indices in the batch to calculate percentages
        total_indices = all_indices.numel()
        
        # Initialize a tensor to store the percentage usage of each code
        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)
        
        # Count the number of occurrences of each index and get their frequency as percentages
        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        # Calculate the percentage
        percentages = (counts.float() / total_indices)
        
        # Populate the corresponding percentages in the codebook_usage_percentage tensor
        codebook_usage_percentage[unique_indices.long()] = percentages
        
        return codebook_usage_percentage
    


    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2) # [bthw, c]
        
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * flat_inputs @ self.embeddings.t() \
                    + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True) # [bthw, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs) # [bthw, ncode]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:]) # [b, t, h, w, ncode]

        embeddings = F.embedding(encoding_indices, self.embeddings) # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1) # [b, c, t, h, w]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)
        

        # print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes
            
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
    

# Multi-scale Codebook
from typing import List, Optional, Tuple, Sequence, Union


class ResConvAfterUpsample(nn.Conv3d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3 if quant_resi < 0 else 1
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BCthw):
        return h_BCthw.mul(1-self.resi_ratio) + super().forward(h_BCthw).mul_(self.resi_ratio)


class SharedResConvAfterUpsample(nn.Module):
    def __init__(self, qresi: ResConvAfterUpsample):
        super().__init__()
        self.qresi: ResConvAfterUpsample = qresi
    
    def __getitem__(self, _) -> ResConvAfterUpsample:
        return self.qresi


class ResConvAfterUpsampleList(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> ResConvAfterUpsample:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class ResConvAfterUpsampleModuleList(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> ResConvAfterUpsample:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class MultiScaleCodebook(nn.Module):
    def __init__(self, n_codes, 
                embedding_dim, no_random_restart=False, 
                restart_thres=1.0, usage_sigma=0.99, fp32_quant=False,
                quant_resi = -0.5, share_quant_resi = 4, default_qresi_counts = 10,
                t_patch_nums = (1, 1, 2, 2, 2, 4, 4, 4, 4, 4),
                v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            ):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.register_buffer('codebook_usage', torch.zeros(n_codes))

        self.call_cnt = 0
        self.usage_sigma = usage_sigma

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

        self.fp32_quant = fp32_quant

        # quant resi

        self.t_patch_nums = t_patch_nums
        self.v_patch_nums = v_patch_nums
        self.quant_resi_ratio = quant_resi

        if share_quant_resi == 1:   # args.qsr
            self.quant_resi = SharedResConvAfterUpsample(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        elif share_quant_resi == 0:
            self.quant_resi = ResConvAfterUpsampleModuleList([(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        else:
            self.quant_resi = ResConvAfterUpsampleList(nn.ModuleList([(ResConvAfterUpsample(embedding_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.z_interplote_down = 'area'
        self.z_interplote_up = 'trilinear'



    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        d = y.shape[0]
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
    

    def calculate_batch_codebook_usage_percentage(self, batch_encoding_indices):
        # Flatten the batch of encoding indices into a single 1D tensor
        all_indices = batch_encoding_indices.flatten()
        
        # Obtain the total number of encoding indices in the batch to calculate percentages
        total_indices = all_indices.numel()
        
        # Initialize a tensor to store the percentage usage of each code
        codebook_usage_percentage = torch.zeros(self.n_codes, device=all_indices.device)
        
        # Count the number of occurrences of each index and get their frequency as percentages
        unique_indices, counts = torch.unique(all_indices, return_counts=True)
        # Calculate the percentage
        percentages = (counts.float() / total_indices)
        
        # Populate the corresponding percentages in the codebook_usage_percentage tensor
        codebook_usage_percentage[unique_indices.long()] = percentages
        
        return codebook_usage_percentage
    


    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)

        # 永远维持THW的结构，差最近邻时候flat，然后会进行quant_res
        B, C, T, H, W = z.shape

        z_no_grad = z.detach()
        accu_h = torch.zeros_like(z_no_grad)


        if self.training:
            all_flat_inputs, all_encode_onehot = [], []
        
        commitment_loss = 0.0
        scale_num = len(self.v_patch_nums)
        ms_encoding_indices = []


        with torch.cuda.amp.autocast(enabled=False):
            
            for si, (tpn, pn) in enumerate(zip(self.t_patch_nums, self.v_patch_nums)):
                tpn = min(tpn, T) 

                # latents
                rest_z = z_no_grad - accu_h.data

                if si != scale_num - 1: # z进行下采样
                    rest_z = F.interpolate(rest_z, size=(tpn, pn, pn), mode=self.z_interplote_down)
                
                z_NC =  rest_z.permute(0, 2, 3, 4, 1).reshape(-1, C)

                # 这个尺度的 rest_z 与 codebook的 distances
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embeddings.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embeddings.t(), alpha=-2, beta=1)  
                
                # 转成离散ids
                encoding_indices = torch.argmin(d_no_grad, dim=1)
                encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(z_NC) # [bthw, ncode]
                encoding_indices = encoding_indices.view(rest_z.shape[0], *rest_z.shape[2:]) # [b, t, h, w, ncode]

                ms_encoding_indices.append(encoding_indices)

                # id转回连续，用h_表述
                h_BTHWC = F.embedding(encoding_indices, self.embeddings)    # [b, t, h, w, c]
                h_BCTHW = h_BTHWC.permute(0, 4, 1, 2, 3).contiguous()    # [b, c, t, h, w]

                # up & quant resi
                                
                h_BCTHW = F.interpolate(h_BCTHW, size=(T, H, W), mode=self.z_interplote_up).contiguous()

                # 加一个quant resi做卷积运算
                quant_head = si / max(1, (scale_num - 1))
                h_BCTHW = self.quant_resi[quant_head](h_BCTHW)

                # h累加
                accu_h = accu_h + h_BCTHW

                commitment_loss += 0.25 * F.mse_loss(accu_h, z.detach())   # 0.25是一个beta

                if self.training:
                    all_flat_inputs.append(z_NC)
                    all_encode_onehot.append(encode_onehot)

        if self.training:

            encode_onehot = torch.cat(all_encode_onehot, dim=0)
            flat_inputs = torch.cat(all_flat_inputs, dim=0)

            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        commitment_loss *= 1.0 / scale_num
        embeddings_st = (accu_h - z_no_grad).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        try:
            usage = self.calculate_batch_codebook_usage_percentage(encoding_indices)
        except:
            usage = torch.zeros(self.n_codes, device=encoding_indices.device)
        

        # print(usage.shape, torch.zeros(self.n_codes).shape)

        if self.call_cnt == 0:
            self.codebook_usage.data = usage
        else:
            self.codebook_usage.data = self.usage_sigma * self.codebook_usage.data + (1 - self.usage_sigma) * usage

        self.call_cnt += 1
        # avg_distribution = self.codebook_usage.data.sum() / self.n_codes
        avg_usage = (self.codebook_usage.data > (1/self.n_codes)).sum() / self.n_codes

        # print(f"training: {embeddings_st.size()=}, {encoding_indices.size()=}")
        # for idx, en_idx in enumerate(ms_encoding_indices):
        #     print(f"{idx=}, {en_idx.size()=}", flush=True)
            
        return dict(embeddings=embeddings_st, encodings=ms_encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity, avg_usage=avg_usage, batch_usage=usage)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

