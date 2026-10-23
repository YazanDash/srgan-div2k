import os
import torch
import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DIV2KDataset(Dataset):
    """
    Dataset loader for paired High-Resolution (HR) and Low-Resolution (LR) images.
    Applies synchronized random cropping and data augmentation.
    """

    def __init__(self, hr_dir: str, lr_dir: str, hr_crop_size: int = 128, scale_factor: int = 4):
        super().__init__()
        self.hr_crop_size = hr_crop_size
        self.scale_factor = scale_factor
        self.lr_crop_size = hr_crop_size // scale_factor

        # Load sorted image file paths
        hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))

        if not hr_files or not lr_files:
            raise ValueError(f"No PNG images found in directories:\nHR: {hr_dir}\nLR: {lr_dir}")

        self.hr_paths = []
        self.lr_paths = []

        # FIX 1: Corrected string split typo (.spli -> .split)
        lr_dict = {os.path.basename(p).split("x")[0].split(".")[0]: p for p in lr_files}

        for hr_p in hr_files:
            base_id = os.path.basename(hr_p).split(".")[0]
            if base_id in lr_dict:
                self.hr_paths.append(hr_p)
                self.lr_paths.append(lr_dict[base_id])

        assert len(self.hr_paths) > 0, "No matching HR/LR file pairs found!"

    def __len__(self):
        return len(self.hr_paths) * 10

    def _sync_transform(self, hr_img: Image.Image, lr_img: Image.Image):
        # 1. Get image dimensions
        w_lr, h_lr = lr_img.size

        if w_lr < self.lr_crop_size or h_lr < self.lr_crop_size:
            raise ValueError(
                f"Image size ({w_lr}x{h_lr}) is smaller than requested LR crop size ({self.lr_crop_size})."
            )

        # 2. Synchronized Random Crop
        x_lr = random.randint(0, w_lr - self.lr_crop_size)
        y_lr = random.randint(0, h_lr - self.lr_crop_size)

        x_hr = x_lr * self.scale_factor
        y_hr = y_lr * self.scale_factor

        lr_crop = lr_img.crop((x_lr, y_lr, x_lr + self.lr_crop_size, y_lr + self.lr_crop_size))
        hr_crop = hr_img.crop((x_hr, y_hr, x_hr + self.hr_crop_size, y_hr + self.hr_crop_size))

        # 3. Synchronized Horizontal Flip (50% chance)
        if random.random() > 0.5:
            hr_crop = TF.hflip(hr_crop)
            lr_crop = TF.hflip(lr_crop)

        # 4. Synchronized Random 90-degree Rotation (50% chance)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_crop = TF.rotate(hr_crop, angle)
            lr_crop = TF.rotate(lr_crop, angle)

        # FIX 2: Convert to Tensor AFTER applying augmentations
        hr_tensor = TF.to_tensor(hr_crop)
        lr_tensor = TF.to_tensor(lr_crop)

        # 5. Normalize from [0, 1] -> [-1, 1] for Tanh Generator compatibility
        hr_tensor = TF.normalize(hr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        lr_tensor = TF.normalize(lr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return lr_tensor,hr_tensor


    def __getitem__(self, idx: int):
        idx = idx % len(self.hr_paths)

        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")

        return self._sync_transform(hr_img, lr_img)

    # Removed 'self' parameter from staticmethod
    @staticmethod
    def visualization_norm(hr_img: torch.Tensor, lr_img: torch.Tensor):
        hr_vis = (hr_img + 1.0) / 2.0
        lr_vis = (lr_img + 1.0) / 2.0
        return hr_vis, lr_vis

    @staticmethod
    def to_imagenet(x: torch.Tensor) -> torch.Tensor:
        x_01 = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x_01 - mean) / std


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    HR_DIR = "DIV2K_train_HR/DIV2K_train_HR"
    LR_DIR = "DIV2K_train_LR_bicubic_X4/X4"

    dataset = DIV2KDataset(
        hr_dir=HR_DIR,
        lr_dir=LR_DIR,
        hr_crop_size=128,  # Yields 128x128 HR patches
        scale_factor=4  # Yields 32x32 LR patches
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    for hr_batch, lr_batch in dataloader:
        print(f"Matched Image Pairs : {len(dataset)}")
        print(f"HR Batch Shape      : {hr_batch.shape}")  # Expected: [16, 3, 128, 128]
        print(f"LR Batch Shape      : {lr_batch.shape}")  # Expected: [16, 3, 32, 32]
        print(f"HR Value Range      : [{hr_batch.min():.2f}, {hr_batch.max():.2f}]")  # Expected: [-1.00, 1.00]
        print(f"LR Value Range      : [{lr_batch.min():.2f}, {lr_batch.max():.2f}]")  # Expected: [-1.00, 1.00]
        break

class DIV2KValDataset(Dataset):
    """
    Validation Dataset: Deterministic center crop with zero random augmentations.
    """
    def __init__(self, hr_dir: str, lr_dir: str, hr_crop_size: int = 256, scale_factor: int = 4):
        super().__init__()
        self.hr_crop_size = hr_crop_size
        self.scale_factor = scale_factor
        self.lr_crop_size = hr_crop_size // scale_factor

        hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))

        lr_dict = {os.path.basename(p).split("x")[0].split(".")[0]: p for p in lr_files}
        self.hr_paths = []
        self.lr_paths = []

        for hr_p in hr_files:
            base_id = os.path.basename(hr_p).split(".")[0]
            if base_id in lr_dict:
                self.hr_paths.append(hr_p)
                self.lr_paths.append(lr_dict[base_id])

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx: int):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        lr_img = Image.open(self.lr_paths[idx]).convert("RGB")

        # 1. Deterministic Center Crop
        hr_crop = TF.center_crop(hr_img, output_size=[self.hr_crop_size, self.hr_crop_size])
        lr_crop = TF.center_crop(lr_img, output_size=[self.lr_crop_size, self.lr_crop_size])

        # 2. Convert to Tensor & Normalize [-1, 1]
        hr_tensor = TF.to_tensor(hr_crop)
        lr_tensor = TF.to_tensor(lr_crop)

        hr_tensor = TF.normalize(hr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        lr_tensor = TF.normalize(lr_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return lr_tensor, hr_tensor