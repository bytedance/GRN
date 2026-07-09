import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos) if type(self.videos) == list else self.videos.shape[0]

    def __getitem__(self, idx):
        video = self.videos[idx]
        if isinstance(video, str):
            video = torch.load(video, weights_only=True)
        return video

def build_dataloader(videos1, videos2):
    dataset1 = VideoDataset(videos1)
    dataset2 = VideoDataset(videos2)
    assert len(dataset1) == len(dataset2)

    dataloader1 = DataLoader(dataset1, batch_size=1, num_workers=8, shuffle=False)
    dataloader2 = DataLoader(dataset2, batch_size=1, num_workers=8, shuffle=False)

    return dataloader1, dataloader2


