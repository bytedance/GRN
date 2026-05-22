import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips
from .utils import build_dataloader

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='vgg', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def calculate_lpips(videos1, videos2, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    print("calculate_lpips...")

    lpips_results = {}
    dataloader1, dataloader2 = build_dataloader(videos1, videos2)
    for video1, video2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        # get a video [timestamps, channel, h, w]
        
        assert video1.shape == video2.shape
        video1 = video1.squeeze(0) * 2 - 1
        video2 = video2.squeeze(0) * 2 - 1

        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            
            loss_fn.to(device)

            # calculate lpips of a video
            value = np.array(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
            if clip_timestamp not in lpips_results:
                lpips_results[clip_timestamp] = value
            else:
                lpips_results[clip_timestamp] += value

    for clip_timestamp in range(len(video1)):
        lpips_results[clip_timestamp] /= len(dataloader1)

    result = {
        "value": lpips_results,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json
    result = calculate_lpips(videos1, videos2, device)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()