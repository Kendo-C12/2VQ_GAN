import os
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

import torchvision.utils as vutils

class VideoClipDataset(Dataset):
    def __init__(self, video_dir, clip_len=8, resize=(128, 128)):
        """
        video_dir: folder containing *.mp4 videos
        clip_len: number of frames per clip
        resize: output frame size (H, W)
        """
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.resize = resize

        self.transform = T.Compose([
            T.ToTensor(),                         # HWC uint8 → CHW float
            T.Resize(resize),
        ])

        # List all mp4 files
        self.videos = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            # if f.endswith(".mov")
        ]
        if len(self.videos) == 0:
            raise Exception("No .mp4 videos found!")

        # Pre-scan video lengths (in frames)
        self.video_frame_counts = []
        for path in self.videos:
            cap = cv2.VideoCapture(path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_frame_counts.append(count)
            cap.release()

    def __len__(self):
        # Each **clip_len** frames count as one sample
        return sum(fc // self.clip_len for fc in self.video_frame_counts)

    def __getitem__(self, idx):
        """
        Randomly pick a video, then randomly pick clip_len consecutive frames.
        """
        # 1. choose which video to sample from
        vid_idx = random.randint(0, len(self.videos) - 1)
        path = self.videos[vid_idx]
        total_frames = self.video_frame_counts[vid_idx]

        # 2. choose a random starting point
        max_start = total_frames - self.clip_len - 1
        start = random.randint(0, max_start)

        # 3. load frames
        cap = cv2.VideoCapture(path)
        frames = []

        # jump to starting point
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for _ in range(self.clip_len):
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)   # (3,H,W)
            frames.append(frame)

        cap.release()

        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
        return frames

    def train_test_split(self, split_ratio=0.8):
        """
        Split dataset into train and test subsets.
        """
        total_videos = len(self.videos)
        split_idx = int(total_videos * split_ratio)

        train_dataset = VideoClipDataset(
            video_dir=self.video_dir,
            clip_len=self.clip_len,
            resize=self.resize
        )
        test_dataset = VideoClipDataset(
            video_dir=self.video_dir,
            clip_len=self.clip_len,
            resize=self.resize
        )

        train_dataset.videos = self.videos[:split_idx]
        train_dataset.video_frame_counts = self.video_frame_counts[:split_idx]

        test_dataset.videos = self.videos[split_idx:]
        test_dataset.video_frame_counts = self.video_frame_counts[split_idx:]

        return train_dataset, test_dataset