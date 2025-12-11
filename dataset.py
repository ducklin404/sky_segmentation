import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pickle

# Dataset
class SkyPatchDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        patches_per_image=3,  
        img_size=224,
        resize_short=512,
        min_fg_ratio=0.01,
        crop_tries=10          
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.patches_per_image = patches_per_image
        self.resize_short = resize_short
        self.min_fg_ratio = min_fg_ratio
        self.crop_tries = crop_tries
        

        # auto find cache folder (for pre-resized images)
        parent = os.path.dirname(os.path.dirname(image_dir))
        self.pickle_path = os.path.join(parent, "image_file_list.pkl")
        cache_img_dir = os.path.join(parent, "cache_resized", "images")
        cache_mask_dir = os.path.join(parent, "cache_resized", "labels")
        self.use_resized_cache = os.path.exists(cache_img_dir) and os.path.exists(cache_mask_dir)
        if self.use_resized_cache:
            self.image_dir = cache_img_dir
            self.mask_dir = cache_mask_dir
            print(f"Using pre-resized cache: {self.image_dir}  {self.mask_dir}")

        # try loading existing pickle
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, "rb") as f:
                    self.image_files = pickle.load(f)
                print(f"Loaded cached filename list: {len(self.image_files)} files")
            except Exception as e:
                print(f"Failed to load pickle, rebuilding list. Reason: {e}")
                self.image_files = self._build_file_list()
                self._save_pickle()
        else:
            print(f"no pickle found at {self.pickle_path}")
            self.image_files = self._build_file_list()
            self._save_pickle()

        self.n_images = len(self.image_files)
        if self.n_images == 0:
            self.__len__ = lambda: 0
        else:
            self.__len__ = lambda: self.n_images * self.patches_per_image

    def __len__(self):
        # overridden above
        return max(0, getattr(self, "n_images", 0) * getattr(self, "patches_per_image", 0))
    
    def _build_file_list(self):
        file_list = []
        image_files = sorted(os.listdir(self.image_dir))

        for name in image_files:
            img_path = os.path.join(self.image_dir, name)
            mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            if img is None or mask is None:
                continue

            file_list.append(name)

        print(f"Scanned and found {len(file_list)} valid image files.")
        return file_list
    
    # save list to pickle
    def _save_pickle(self):
        try:
            with open(self.pickle_path, "wb") as f:
                pickle.dump(self.image_files, f)
            print(f"Saved filename list to {self.pickle_path}")
        except Exception as e:
            print(f"Failed to save pickle: {e}")

    def __getitem__(self, idx):

        # map global idx -> image index
        img_idx = idx // self.patches_per_image
        name = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')

        # load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to read {img_path} or {mask_path}")

        # resize if not using pre-resized cache
        if not self.use_resized_cache:
            h, w = img.shape[:2]
            scale = self.resize_short / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            new_h, new_w = img.shape[:2]

        # pad if needed
        if new_h < self.img_size or new_w < self.img_size:
            pad_h = max(0, self.img_size - new_h)
            pad_w = max(0, self.img_size - new_w)
            top = pad_h // 2
            left = pad_w // 2
            img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=0)
            new_h, new_w = img.shape[:2]

        # content-aware random crop
        last_img_patch = None
        last_mask_patch = None

        for attempt in range(self.crop_tries):
            x = random.randint(0, new_w - self.img_size)
            y = random.randint(0, new_h - self.img_size)

            img_patch = img[y:y + self.img_size, x:x + self.img_size]
            mask_patch = mask[y:y + self.img_size, x:x + self.img_size]

            # ensure binary mask
            m = (mask_patch > 127).astype(np.uint8)
            frac_minority = min(m.mean(), 1.0 - m.mean())

            last_img_patch = img_patch
            last_mask_patch = mask_patch

            if frac_minority >= self.min_fg_ratio:
                break

        # fallback patch if nothing good found
        img_patch = last_img_patch
        mask_patch = last_mask_patch

        # downscale mask to half resolution (3-block UNet style)
        out_size = self.img_size // 2
        mask_patch = cv2.resize(mask_patch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        mask_patch = (mask_patch > 127).astype(np.uint8)

        # HWC -> CHW and convert ranges
        img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.0
        mask_patch = torch.from_numpy(mask_patch).float()

        return img_patch, mask_patch