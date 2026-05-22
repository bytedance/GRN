import numpy as np
import torch
from tqdm import tqdm
import math
from multiprocessing import Pool
from .utils import build_dataloader

def img_psnr(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def trans(x):
    return x

def process_video(video_pair):
    video1, video2 = video_pair
    psnr_results_of_a_video = []
    for clip_timestamp in range(len(video1)):
        # get a img
        # img [timestamps[x], channel, h, w]
        # img [channel, h, w] numpy

        img1 = video1[clip_timestamp].numpy()
        img2 = video2[clip_timestamp].numpy()
        
        # calculate psnr of a video
        psnr_results_of_a_video.append(img_psnr(img1, img2))
    
    return psnr_results_of_a_video
    

def calculate_psnr(videos1, videos2):
    print("calculate_psnr...")

    # videos [batch_size, timestamps, channel, h, w]
    dataloader1, dataloader2 = build_dataloader(videos1, videos2)
    psnr_results = []
    for video1, video2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        video1 = video1.squeeze(0)
        video2 = video2.squeeze(0)
        video_pair = (video1, video2)
        result = process_video(video_pair)
        psnr_results.append(result)

    psnr_results = np.array(psnr_results)
    
    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(len(video1)):
        psnr[clip_timestamp] = np.mean(psnr_results[:,clip_timestamp])
        psnr_std[clip_timestamp] = np.std(psnr_results[:,clip_timestamp])

    result = {
        "value": psnr,
        "value_std": psnr_std,
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)

    import json
    result = calculate_psnr(videos1, videos2)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()