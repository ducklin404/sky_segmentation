import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

# Dataset
class SkyPatchDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        img_size=224,
        output_scale=2,
        resize_short=2240,
        stride=112,
        min_fg_ratio=0.01,
        cache_file=None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.out_size = img_size // output_scale
        self.stride = stride
        self.min_fg_ratio = min_fg_ratio
        self.resize_short = resize_short

        # use cache file if possible
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading dataset cache: {cache_file}")
            with open(cache_file, "rb") as f:
                self.samples = pickle.load(f)
            return

        # build it norammmly if no cache file
        self.samples = []

        image_files = sorted(os.listdir(image_dir))

        for name in image_files:
            img_path = os.path.join(image_dir, name)
            mask_path = os.path.join(mask_dir, name.split('.')[0] + '.png')

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            h, w = img.shape[:2]
            scale = resize_short / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)

            img = cv2.resize(img, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

            for y in range(0, new_h - img_size + 1, stride):
                for x in range(0, new_w - img_size + 1, stride):
                    m_patch = mask[y:y+img_size, x:x+img_size]

                    sky_ratio = (m_patch == 0).mean()
                    nonsky_ratio = (m_patch == 1).mean()

                    if (sky_ratio > min_fg_ratio or nonsky_ratio > min_fg_ratio):
                        self.samples.append((img_path, mask_path, x, y, scale))

        # save cache 
        if cache_file is not None:
            with open(cache_file, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"Saved dataset cache: {cache_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, x, y, scale = self.samples[idx]

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        img = cv2.resize(img, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)

        img_patch = img[y:y+224, x:x+224]
        mask_patch = mask[y:y+224, x:x+224]

        # downscale mask to 112×112
        mask_patch = cv2.resize(
            mask_patch, (112, 112),
            interpolation=cv2.INTER_NEAREST
        )

        img_patch = torch.from_numpy(
            img_patch).permute(2, 0, 1).float() / 255.0
        mask_patch = torch.from_numpy(mask_patch).long()

        return img_patch, mask_patch
