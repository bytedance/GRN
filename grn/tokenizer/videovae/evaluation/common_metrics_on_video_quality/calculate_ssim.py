import numpy as np
import torch
from tqdm import tqdm
import cv2
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from .utils import build_dataloader
 
def ssim(img1, img2):
    _img1, _img2 = img1, img2

    ### original implementation
    # C1 = 0.01 ** 2
    # C2 = 0.03 ** 2
    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    # kernel = cv2.getGaussianKernel(11, 1.5)
    # window = np.outer(kernel, kernel.transpose())
    # mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    # mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    # mu1_sq = mu1 ** 2
    # mu2_sq = mu2 ** 2
    # mu1_mu2 = mu1 * mu2
    # sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    # sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    # sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
    #                                                         (sigma1_sq + sigma2_sq + C2))
    # return ssim_map.mean()
    # res1 = ssim_map.mean()

    ### accelerated implementation
    img1, img2 = torch.from_numpy(_img1), torch.from_numpy(_img2)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Ensure data is float and move to GPU
    img1 = img1.to(torch.float64).cuda()
    img2 = img2.to(torch.float64).cuda()
    
    # Gaussian kernel
    kernel = torch.tensor(cv2.getGaussianKernel(11, 1.5)).to(torch.float64).cuda()
    window = kernel @ kernel.t()
    window = window.unsqueeze(0).unsqueeze(0).cuda()
    
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=0, groups=1)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=0, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1.unsqueeze(0) ** 2, window, padding=0, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) ** 2, window, padding=0, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=0, groups=1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    res2 = ssim_map.mean().item()
    # print(res1-res2)
    return res2
    # return ssim_map.mean().item()
 
 
def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w 
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()                   
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def trans(x):
    return x

def process_video(video_pair):
    video1, video2 = video_pair
    ssim_results_of_a_video = []
    for clip_timestamp in range(len(video1)):
        img1 = video1[clip_timestamp].numpy()
        img2 = video2[clip_timestamp].numpy()
        ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))
    return ssim_results_of_a_video

def calculate_ssim(videos1, videos2):
    print("calculate_ssim...")

    ssim_results = []
    dataloader1, dataloader2 = build_dataloader(videos1, videos2)
    for video1, video2 in tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)):
        video1 = video1.squeeze(0)
        video2 = video2.squeeze(0)
        video_pair = (video1, video2)
        result = process_video(video_pair)
        ssim_results.append(result)

    ssim_results = np.array(ssim_results)

    ssim = {}
    ssim_std = {}

    for clip_timestamp in range(len(video1)):
        ssim[clip_timestamp] = np.mean(ssim_results[:,clip_timestamp])
        ssim_std[clip_timestamp] = np.std(ssim_results[:,clip_timestamp])

    result = {
        "value": ssim,
        "value_std": ssim_std,
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
    device = torch.device("cuda")

    import json
    result = calculate_ssim(videos1, videos2)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()