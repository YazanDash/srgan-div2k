import os
from PIL import Image
from tqdm import tqdm

hr_val_dir = "DIV2K_valid_HR/DIV2K_valid_HR"
lr_val_dir = "DIV2K_valid_LR_bicubic_X4/X4"
os.makedirs(lr_val_dir, exist_ok=True)

hr_files = [f for f in os.listdir(hr_val_dir) if f.endswith(".png")]

print("Generating 100 Low-Resolution Validation Images...")
for filename in tqdm(hr_files):
    hr_path = os.path.join(hr_val_dir, filename)

    # DIV2K official naming convention for X4 downsampling (e.g., 0801x4.png)
    base_name = os.path.splitext(filename)[0]
    lr_filename = f"{base_name}x4.png"
    lr_path = os.path.join(lr_val_dir, lr_filename)

    # Open HR image and downsample 4x using PIL Bicubic kernel
    img = Image.open(hr_path).convert("RGB")
    w, h = img.size
    lr_img = img.resize((w // 4, h // 4), Image.BICUBIC)
    lr_img.save(lr_path)

print(f"Done! Saved 100 LR validation images to '{lr_val_dir}'")