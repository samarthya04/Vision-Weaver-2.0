import os
import math
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# --- CUDA-Safe Multiprocessing for Mamba ---
# Mamba's CUDA kernels can conflict with the default 'fork' on Linux.
# 'spawn' ensures workers have a clean memory space.
try:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
except RuntimeError:
    pass

class ResizeByScale:
    """Efficient callable for scaling factors."""
    def __init__(self, scale):
        self.scale = scale
        self.interpolation = transforms.InterpolationMode.BILINEAR

    def __call__(self, img: Image) -> Image:
        new_size = (int(img.height * self.scale), int(img.width * self.scale))
        return transforms.functional.resize(img, new_size, interpolation=self.interpolation)

class PairedImagesDataset(Dataset):
    """Memory-mapped ready dataset for paired Super-Resolution images."""
    def __init__(self, lr_dir: Path, hr_dir: Path, transform_hr=None, transform_lr=None):
        self.lr_dir, self.hr_dir = lr_dir, hr_dir
        self.transform_hr, self.transform_lr = transform_hr, transform_lr
        
        # Performance: Pre-filter file list to avoid filesystem thrashing
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.tif')
        self.file_pairs = [
            f for f in sorted(os.listdir(lr_dir)) if f.lower().endswith(valid_exts)
        ]

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.file_pairs[idx]
        
        # Load and convert to RGB in one step to minimize PIL overhead
        img_lr = Image.open(self.lr_dir / filename).convert("RGB")
        img_hr = Image.open(self.hr_dir / filename).convert("RGB")

        if self.transform_hr: img_hr = self.transform_hr(img_hr)
        if self.transform_lr: img_lr = self.transform_lr(img_lr)

        return img_lr, img_hr

class PairedImagesDataModule(pl.LightningDataModule):
    """
    Lightning DataModule optimized for GPU throughput.
    Includes Pin-Memory, Persistent Workers, and Vectorized Batching.
    """
    def __init__(self, cfg, lr_dir: Path, hr_dir: Path, batch_size: int = 32, scale: int = None):
        super().__init__()
        self.cfg = cfg
        self.lr_dir, self.hr_dir = lr_dir, hr_dir
        self.batch_size = batch_size
        self.scale = scale if scale is not None else self.cfg.dataset.scale

        # Latent Manifold Normalization (Mean=0.5, Std=0.5)
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        
        if self.cfg.dataset.resize:
            self.transform_hr = transforms.Compose(transform_list)
            self.transform_lr = transforms.Compose([ResizeByScale(self.scale)] + transform_list)
        else:
            self.transform_hr = transforms.Compose(transform_list)
            self.transform_lr = transforms.Compose(transform_list)

    def setup(self, stage=None):
        """Prepares train, val, and test splits with path safety."""
        if stage == "fit" or stage is None:
            self.paired_images_train = PairedImagesDataset(self.lr_dir / "train", self.hr_dir / "train", self.transform_hr, self.transform_lr)
            self.paired_images_val = PairedImagesDataset(self.lr_dir / "val", self.hr_dir / "val", self.transform_hr, self.transform_lr)
        
        elif stage == "train_val_test":
            for s in ["train", "val", "test"]:
                path_lr = Path(str(self.lr_dir).replace("|train_val_test|", s))
                path_hr = Path(str(self.hr_dir).replace("|train_val_test|", s))
                setattr(self, f"paired_images_{s}", PairedImagesDataset(path_lr, path_hr, self.transform_hr, self.transform_lr))
                
        elif stage == "only_test":
            self.paired_images_test = PairedImagesDataset(self.lr_dir, self.hr_dir, self.transform_hr, self.transform_lr)

    def train_dataloader(self):
        """Optimized for non-blocking GPU data transfer."""
        return DataLoader(
            self.paired_images_train,
            batch_size=self.batch_size,
            shuffle=True,
            # Caps workers at 4. Utilizing 'spawn' multiprocessing with many workers 
            # duplicates the Python interpreter memory footprint and causes RAM OOMs.
            num_workers=min(4, os.cpu_count() or 1), 
            collate_fn=self.collate_cropping_fn,
            persistent_workers=True, # Prevents overhead between epochs
            pin_memory=True,         # Faster CPU -> GPU transfer via DMA
            prefetch_factor=2        # Prepares next batches while current is training
        )

    def val_dataloader(self):
        return DataLoader(
            self.paired_images_val, 
            batch_size=self.batch_size, 
            num_workers=min(2, os.cpu_count() or 1), 
            collate_fn=self.collate_padding_fn, 
            persistent_workers=True, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.paired_images_test, 
            batch_size=self.batch_size, 
            num_workers=min(2, os.cpu_count() or 1), 
            collate_fn=self.collate_padding_fn, 
            persistent_workers=True, 
            pin_memory=True
        )

    def collate_cropping_fn(self, batch):
        """Vectorized random cropping to ensure batch consistency (Multiple of 64)."""
        k = 64
        # Calculate optimal crop size for the current batch
        min_h = min(img.size(1) for img, _ in batch) // k * k
        min_w = min(img.size(2) for img, _ in batch) // k * k

        lr_list, hr_list, orig_sizes = [], [], []
        for img_lr, img_hr in batch:
            h, w = img_lr.shape[1], img_lr.shape[2]
            
            # Efficient random selection
            y = np.random.randint(0, h - min_h + 1)
            x = np.random.randint(0, w - min_w + 1)
            
            lr_list.append(img_lr[:, y : y + min_h, x : x + min_w])
            hr_list.append(img_hr[:, y : y + min_h, x : x + min_w])
            orig_sizes.append((w, h))

        return {
            "lr": torch.stack(lr_list), 
            "hr": torch.stack(hr_list), 
            "original_size": orig_sizes
        }

    def collate_padding_fn(self, batch):
        """Zero-pads images for validation/inference to preserve exact aspect ratios."""
        k = 64
        max_h = math.ceil(max(img.size(1) for img, _ in batch) / k) * k
        max_w = math.ceil(max(img.size(2) for img, _ in batch) / k) * k
        
        lr_list, hr_list, p_lr, p_hr = [], [], [], []
        for img_lr, img_hr in batch:
            pad_h, pad_w = max_h - img_lr.size(1), max_w - img_lr.size(2)
            
            # Use F.pad for faster tensor operations
            lr_list.append(F.pad(img_lr, (0, pad_w, 0, pad_h)))
            hr_list.append(F.pad(img_hr, (0, pad_w, 0, pad_h)))
            p_lr.append((img_lr.size(2), img_lr.size(1)))
            p_hr.append((img_hr.size(2), img_hr.size(1)))

        return {
            "lr": torch.stack(lr_list), 
            "hr": torch.stack(hr_list), 
            "padding_data_lr": p_lr, 
            "padding_data_hr": p_hr
        }

def train_val_test_loader(cfg):
    """Factory for loading dataset paths from YAML."""
    name, scale = cfg.dataset.name, cfg.dataset.scale
    
    if name in ["div2k", "realsr_canon", "realsr_nikon"]:
        hr_dir = Path(f"data/{name}/|train_test|/X{scale}/HR")
        lr_dir = Path(f"data/{name}/|train_test|/X{scale}/LR")
        dm = PairedImagesDataModule(cfg, lr_dir, hr_dir, cfg.dataset.batch_size)
        dm.setup(stage="train_test")
    elif name in ["celeb", "imagenet"]:
        hr_dir = Path(f"data/{name}/|train_val_test|/X{scale}/HR")
        lr_dir = Path(f"data/{name}/|train_val_test|/X{scale}/LR")
        dm = PairedImagesDataModule(cfg, lr_dir, hr_dir, cfg.dataset.batch_size)
        dm.setup("train_val_test")
    elif name in ["Set14", "urban100"]:
        hr_dir = Path(f"data/{name}/X{scale}/HR")
        lr_dir = Path(f"data/{name}/X{scale}/LR")
        dm = PairedImagesDataModule(cfg, lr_dir, hr_dir, cfg.dataset.batch_size)
        dm.setup("only_test")
        return None, None, dm.test_dataloader()
    else:
        raise ValueError(f"Dataset '{name}' not recognized.")

    return dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()