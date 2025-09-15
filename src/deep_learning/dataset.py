# src/deep_learning/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TuSimpleDataset(Dataset):
    """
    Loads flattened image/mask pairs from images_dir and masks_dir.
    Expects images named *.jpg and masks named the same base with .png.
    Returns (img_tensor, mask_tensor) where:
      - img_tensor: float32 [3, H, W], values in [0,1]
      - mask_tensor: float32 [1, H, W], binary {0,1}
    """
    def __init__(self, images_dir, masks_dir, img_size=(512, 288)):
        self.images_dir = os.path.normpath(images_dir)
        self.masks_dir = os.path.normpath(masks_dir)
        self.img_size = img_size  # (width, height)

        all_images = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(".jpg")])
        self.samples = []
        for img_name in all_images:
            mask_name = os.path.splitext(img_name)[0] + ".png"
            img_path = os.path.join(self.images_dir, img_name)
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                # skip images without masks
                continue

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {images_dir} and {masks_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # read
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask {mask_path}")

        # resize
        img = cv2.resize(img, self.img_size)  # img_size is (w,h)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # normalize & to tensor
        img = img.astype("float32") / 255.0
        mask = (mask > 0).astype("float32")

        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()

        return img, mask
