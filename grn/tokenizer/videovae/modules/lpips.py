"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import os, hashlib
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from videovae.utils.misc import data_prefix_manager, set_tf32_flags

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def build_lpips_model(args):
    if hasattr(args, 'lpips_model'):
        if args.lpips_model == "vgg":
            image_perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval() 
            video_perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval() 
        elif args.lpips_model == "resnet50":
            image_perceptual_model = ResNet50LPIPS().eval()
            video_perceptual_model = ResNet50LPIPS().eval()
        elif args.lpips_model == "swin3d_t":
            image_perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval() 
            video_perceptual_model = Swin3DTLPIPS(perceptual_layers=args.video_perceptual_layers).eval()
        else:
            raise NotImplementedError
        for model in [image_perceptual_model, video_perceptual_model]:
            for param in model.parameters():
                param.requires_grad = False
    else:
        image_perceptual_model = video_perceptual_model = None
    return image_perceptual_model, video_perceptual_model

class ResNet50LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.lpips_net = nn.Sequential(*(list(resnet50.children())[:-2]))
        self.lpips_loss = nn.MSELoss()
    
    def forward(self, input, target):
        return self.lpips_loss(self.lpips_net(input), self.lpips_net(target),)

# Define a function to hook into each SwinTransformerBlockV2
def get_features(outputs, hooks, x):
    """Get features at each hook"""
    hooks.append(x)

# Custom feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, model, perceptual_layers=[]):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.hooks = []

        # Register hooks at each SwinTransformerBlockV2 layer
        for name, module in self.model.named_modules():
            if name in perceptual_layers:
                module.register_forward_hook(lambda module, input, output: get_features(output, self.hooks, input))

    def forward(self, x):
        self.hooks = []
        _ = self.model(x)
        return self.hooks

class Swin3DTLPIPS(nn.Module):
    def __init__(self, perceptual_layers=[]):
        super().__init__()
        swin_v2_s_model = models.video.swin3d_t(weights=models.video.Swin3D_T_Weights.DEFAULT)
        self.feature_extractor = FeatureExtractor(swin_v2_s_model, perceptual_layers=perceptual_layers)
        self.lpips_loss = nn.MSELoss()
    
    def forward(self, rec_video, gt_video):
        rec_features = self.feature_extractor(rec_video)
        gt_features = self.feature_extractor(gt_video)
        loss = 0.
        for rec_f, gt_f in zip(rec_features, gt_features):
            mse_loss = self.lpips_loss(rec_f[0], gt_f[0])
            loss += mse_loss
        loss = loss / len(rec_features)
        return loss

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True, upcast_tf32=False):
        super().__init__()
        self.upcast_tf32 = upcast_tf32
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name is not "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True), strict=False)
        return model

    def forward(self, input, target):
        with set_tf32_flags(not self.upcast_tf32):
            in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
            outs0, outs1 = self.net(in0_input), self.net(in1_input)
            feats0, feats1, diffs = {}, {}, {}
            lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            for kk in range(len(self.chns)):
                feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
                
            res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
            val = res[0]
            for l in range(1, len(self.chns)):
                # print(res[l].shape)
                val += res[l]
            
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        # load locally
        assert pretrained == True
        vgg_model = models.vgg16()
        vgg_model.load_state_dict(torch.load(data_prefix_manager("checkpoints/vgg16-397923af.pth"), weights_only=True))
        vgg_pretrained_features = vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    # norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    norm_factor = x.norm(p=2, dim=1, keepdim=True)
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)
