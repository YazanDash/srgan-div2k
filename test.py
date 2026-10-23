import argparse
import time
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image with Auto-Downsampling')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, required=True, help='high-resolution test image path')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--num_res_blocks', default=5, type=int, help='must match training')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

# 1. Load Model
model = Generator(UPSCALE_FACTOR, num_res_blocks=opt.num_res_blocks).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('checkpoints/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('checkpoints/' + MODEL_NAME, map_location=lambda storage, loc: storage))

# 2. Load High-Resolution Image and Downsample
hr_image = Image.open(IMAGE_NAME).convert("RGB")
w, h = hr_image.size

# Ensure dimensions are divisible by the upscale factor
lr_w, lr_h = w // UPSCALE_FACTOR, h // UPSCALE_FACTOR
hr_image = hr_image.resize((lr_w * UPSCALE_FACTOR, lr_h * UPSCALE_FACTOR), Image.BICUBIC)
w, h = hr_image.size

# Downsample to create LR input using Bicubic interpolation
lr_image = hr_image.resize((w // UPSCALE_FACTOR, h // UPSCALE_FACTOR), Image.BICUBIC)

# Save bicubic baseline for comparison
output_dir = f"out_srf_{UPSCALE_FACTOR}"
os.makedirs(output_dir, exist_ok=True)
base_image_name = os.path.basename(IMAGE_NAME)
lr_image.resize((w, h), Image.BICUBIC).save(os.path.join(output_dir, f"bicubic_{base_image_name}"))
lr_image.save(os.path.join(output_dir, f"lr_{base_image_name}"))
# 3. Preprocess LR image tensor
image_tensor = ToTensor()(lr_image).unsqueeze(0)
image_tensor = image_tensor * 2.0 - 1.0  # Normalize to [-1, 1]

if TEST_MODE:
    image_tensor = image_tensor.cuda()

# 4. Inference
with torch.no_grad():
    start = time.perf_counter()
    out = model(image_tensor)
    elapsed = time.perf_counter() - start

print(f'Inference time: {elapsed:.4f}s')

# 5. Denormalize output from [-1, 1] -> [0, 1]
out = (out + 1.0) / 2.0
out = torch.clamp(out, 0.0, 1.0)

# 6. Save SR Result
out_img = ToPILImage()(out[0].cpu())
output_filename = os.path.join(output_dir, f"sr_{base_image_name}")
out_img.save(output_filename)

print(f'Saved SR result to {output_filename}')
print(f'Saved Bicubic baseline to {os.path.join(output_dir, f"bicubic_{base_image_name}")}')