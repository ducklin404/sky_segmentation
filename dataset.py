import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import random


# Dataset
class SkyPatchDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        patches_per_image=3,  
        img_size=224,
        resize_short=2240,
        stride=112,
        min_fg_ratio=0.01,
        cache_file=None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.patches_per_image = patches_per_image
        self.stride = stride
        self.resize_short = resize_short
        self.min_fg_ratio = min_fg_ratio

        # auto find cache folder
        parent = os.path.dirname(image_dir)
        cache_img_dir = os.path.join(parent, "cache_resized", "images")
        cache_mask_dir = os.path.join(parent, "cache_resized", "labels")
        self.use_resized_cache = os.path.exists(cache_img_dir) and os.path.exists(cache_mask_dir)
        if self.use_resized_cache:
            self.image_dir = cache_img_dir
            self.mask_dir = cache_mask_dir
            print(f"Using pre-resized cache: {self.image_dir}  {self.mask_dir}")


        # if cache_file provided and exists, load coordinates / samples index.
        # format: dict with keys "image_files" and "coords_per_image"
        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading dataset cache: {cache_file}")
            with open(cache_file, "rb") as f:
                loaded = pickle.load(f)
                
            if isinstance(loaded, dict):
                self.image_files = loaded["image_files"]
                self.coords_per_image = loaded["coords_per_image"]
            else:
                raise RuntimeError("Unknown cache_file format")
        else:
            # build it normally if no cache file
            self.image_files = []
            self.coords_per_image = []

            image_files = sorted(os.listdir(self.image_dir))

            for name in image_files:
                img_path = os.path.join(self.image_dir, name)
                mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')

                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, 0)

                if img is None or mask is None:
                    # skip missing or unreadable images
                    continue

                # resize
                if self.use_resized_cache:
                    new_h, new_w = img.shape[:2]
                    scale = 1.0
                    img_r = img
                    mask_r = mask
                else:
                    h, w = img.shape[:2]
                    scale = resize_short / min(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_r = cv2.resize(img, (new_w, new_h))
                    mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # ensure mask is 0/1
                mask_bin = (mask_r > 127).astype(np.uint8)

                coords = []
                for y in range(0, new_h - img_size + 1, stride):
                    for x in range(0, new_w - img_size + 1, stride):
                        m_patch = mask_bin[y:y+img_size, x:x+img_size]

                        sky_ratio = (m_patch == 0).mean()
                        nonsky_ratio = (m_patch == 1).mean()

                        if (sky_ratio > min_fg_ratio or nonsky_ratio > min_fg_ratio):
                            coords.append((x, y, scale))

                if len(coords) == 0:
                    # fallback: allow center patch if no coords passed threshold
                    cx = max(0, (new_w - img_size) // 2)
                    cy = max(0, (new_h - img_size) // 2)
                    coords = [(cx, cy, scale)]

                self.image_files.append(name)
                self.coords_per_image.append(coords)

            # save cache as dict of image_files + coords_per_image for faster rebuilds
            if cache_file is not None:
                cache_data = {
                    "image_files": self.image_files,
                    "coords_per_image": self.coords_per_image
                }
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                print(f"Saved dataset cache: {cache_file}")

        # final length: each image contributes patches_per_image samples per epoch
        self.n_images = len(self.image_files)

        if self.n_images == 0:
            self.__len__ = lambda: 0
        else:
            self.__len__ = lambda: self.n_images * self.patches_per_image

    def __len__(self):
        # this will be overridden but keep signature
        return max(0, getattr(self, "n_images", 0) * getattr(self, "patches_per_image", 0))

    def __getitem__(self, idx):

        # map global idx -> image index
        img_idx = idx // self.patches_per_image
        name = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')

        # pick a random coord for this image (randomized each epoch)
        coords = self.coords_per_image[img_idx]
        x, y, scale = random.choice(coords)

        # load image & mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        if not self.use_resized_cache:
            # resize according to precomputed scale
            h, w = img.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            # use cache
            new_h, new_w = img.shape[:2]

        # clamp in case coordinates are near edge after rounding
        x = min(max(0, x), max(0, new_w - self.img_size))
        y = min(max(0, y), max(0, new_h - self.img_size))

        img_patch = img[y:y + self.img_size, x:x + self.img_size]
        mask_patch = mask[y:y + self.img_size, x:x + self.img_size]

        # downscale mask to half resolution (3 block u net)
        out_size = self.img_size // 2
        mask_patch = cv2.resize(mask_patch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        # ensure mask is binary 0/1 (0 = sky, 1 = nonsky)
        mask_patch = (mask_patch > 127).astype(np.uint8)

        # HWC -> CHW and convert ranges
        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        # return float mask for BCEWithLogitsLoss
        mask_patch = torch.from_numpy(mask_patch).float()

        return img_patch, mask_patch
