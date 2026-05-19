from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = sorted(list(self.image_dir.glob("*.png")))
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = np.expand_dims(image, axis=0)   # [1, H, W]
        mask = np.expand_dims(mask, axis=0)     # [1, H, W]

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)