import numpy as np
import random
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from videovae.utils.misc import set_tf32_flags
from videovae.modules.normalization import Normalize


class DiscriminatorPool:
    def __init__(self, pool_size):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.images.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.pool_size - 1)
                    tmp = self.images[i].clone()
                    self.images[i] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.stack(return_images)

class ImageDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(use_blur=args.disc_use_blur, downsample_base=args.disc_stylegan_downsample_base)
        elif args.disc_type == "spectralgan":
            print("using NLayerSpectralDiscriminator")
            self.discriminator = NLayerSpectralDiscriminator()
        else:
            self.discriminator = NLayerDiscriminator() # PatchGAN, default; args.disc_pool=no, default; temporal_compress=yes, default;
        self.disc_pool = args.disc_pool
        if args.disc_pool == "yes":
            self.real_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
            self.fake_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
    
    def forward(self, x, pool_name=None):
        if pool_name and self.disc_pool == "yes":
            assert pool_name in ["real", "fake"]
            if pool_name == "real":
                x = self.real_pool.query(x)
            elif pool_name == "fake":
                x = self.fake_pool.query(x)
        return self.discriminator(x)

class VideoDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(conv_type="3d", use_blur=args.disc_use_blur, downsample_base=args.disc_stylegan_downsample_base)
            # self.discriminator = MagvitDiscriminator(args.image_channels, apply_blur=args.apply_blur, apply_noise=args.apply_noise, model_type="3d", use_checkpoint=args.use_checkpoint, version=args.disc_version, norm_type=args.norm_type)
        elif args.disc_type == "spectralgan":
            print("using NLayerSpectralDiscriminator3D")
            self.discriminator = NLayerSpectralDiscriminator3D(temporal_compress=args.disc_temporal_compress)
        else: # PatchGAN, default; args.disc_pool=no, default; temporal_compress=yes, default;
            self.discriminator = NLayerDiscriminator3D(temporal_compress=args.disc_temporal_compress)
            # self.discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers, args.norm_type, use_sigmoid=args.sigmoid_in_disc, activation=args.activation_in_disc, apply_blur=args.apply_blur, apply_noise=args.apply_noise, upcast_tf32=args.upcast_tf32)
        self.disc_pool = args.disc_pool
        if args.disc_pool == "yes":
            self.real_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
            self.fake_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
    
    def forward(self, x, pool_name=None):
        if pool_name and self.disc_pool == "yes":
            assert pool_name in ["real", "fake"]
            if pool_name == "real":
                x = self.real_pool.query(x)
            elif pool_name == "fake":
                x = self.fake_pool.query(x)
        return self.discriminator(x)


class NLayerSpectralDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerSpectralDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

class NLayerSpectralDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, temporal_compress="yes"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerSpectralDiscriminator3D, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1,2,2), padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        _stride = 2 if temporal_compress == "yes" else (1,2,2)
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.utils.spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=_stride, padding=padw, bias=True)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            torch.nn.utils.spectral_norm(nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv3d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_type = "batch"
        use_bias = norm_type != "batch"

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                Normalize(ndf * nf_mult, norm_type=norm_type),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            Normalize(ndf * nf_mult, norm_type=norm_type),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, temporal_compress="yes"):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        norm_type = "batch"
        use_bias = norm_type != "batch"

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=(1,2,2), padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        _stride = 2 if temporal_compress == "yes" else (1,2,2)
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=_stride, padding=padw, bias=use_bias),
                Normalize(ndf * nf_mult, norm_type=norm_type),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            Normalize(ndf * nf_mult, norm_type=norm_type),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv3d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

class ResBlockDown(nn.Module):
    def __init__(self, ic, oc, model_type="2d"):
        super(ResBlockDown, self).__init__()
        assert model_type in ["2d", "3d"]
        activation_func = nn.LeakyReLU(0.2, True)

        if model_type == "2d":
            self.branch1 = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1),
                activation_func,
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
                activation_func,
            )
            self.branch2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(ic, oc, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise NotImplementedError
    
    def forward(self, x):
        return self.branch1(x) + self.branch2(x)

########################################
#             StyleGAN                 #
########################################
class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, channel_multiplier=1, image_size=256, conv_type="2d", use_blur=True, downsample_base=2):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        assert conv_type in ["2d", "3d"]
        conv = nn.Conv2d if conv_type == "2d" else nn.Conv3d
        
        log_size = int(math.log(image_size, 2))
        in_channel = channels[image_size]

        blocks = [conv(input_nc, in_channel, 3, padding=1), leaky_relu()]
        for i in range(log_size, downsample_base, -1):
            out_channel = channels[2 ** (i - 1)]
            blocks.append(DiscriminatorBlock(in_channel, out_channel, conv_type=conv_type, use_blur=use_blur))
            in_channel = out_channel
        self.blocks = nn.ModuleList(blocks)

        self.final_conv = nn.Sequential(
            conv(in_channel, channels[4], 3, padding=1),
            leaky_relu(),
        )
        self.final_linear = nn.Sequential(
            conv(channels[4], 1, 1, padding=0)
        )
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = self.final_linear(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, conv_type="2d", use_blur=True):
        super().__init__()
        assert conv_type in ["2d", "3d"]
        conv = nn.Conv2d if conv_type == "2d" else nn.Conv3d
        if conv_type == "2d":
            conv = nn.Conv2d
            stride = 2 if downsample else 1
        elif conv_type == "3d":
            conv = nn.Conv3d
            stride = (1,2,2) if downsample else (1,1,1)

        self.conv_res = conv(input_channels, filters, 1, stride = stride)

        self.net = nn.Sequential(
            conv(input_channels, filters, 3, padding=1),
            leaky_relu(),
            conv(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur() if use_blur else nn.Identity(),
            conv(filters, filters, 3, padding = 1, stride = stride)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    
    def forward(self, x):
        is_image = x.ndim == 4
        if not is_image:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        from kornia.filters import filter2d
        x = filter2d(x, f, normalized=True)
        if not is_image:
            x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return x

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def exists(val):
    return val is not None
