import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm

from model import Generator

EXTS = (".png", ".jpg", ".jpeg", ".bmp")

parser = argparse.ArgumentParser(description="Evaluate bicubic vs. SR model on a folder of HR images")
parser.add_argument("--data_dir", required=True, help="Folder with HR images (e.g. Set5/)")
parser.add_argument("--checkpoint", required=True, help="Path to generator .pth file")
parser.add_argument("--num_res_blocks", type=int, default=5, help="Must match training")
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--output_dir", default="eval_out", help="Where to save images and results.csv")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
parser.add_argument("--no_save_images", action="store_true", help="Only compute metrics")
opt = parser.parse_args()


def rgb_to_y(img_uint8):
    """HxWx3 uint8 -> Y channel (BT.601, range 16..235), float64."""
    x = img_uint8.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def y_metrics(pred, target, border):
    """Y-channel PSNR and SSIM with `border` pixels cropped on every side."""
    py, ty = rgb_to_y(pred), rgb_to_y(target)
    if border > 0:
        py = py[border:-border, border:-border]
        ty = ty[border:-border, border:-border]
    mse = np.mean((py - ty) ** 2)
    psnr = 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-10))
    ssim = ssim_fn(py, ty, data_range=255.0)
    return psnr, ssim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    s = opt.scale

    model = Generator(s, num_res_blocks=opt.num_res_blocks).to(device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(opt.data_dir) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit(f"No images found in {opt.data_dir}")

    os.makedirs(opt.output_dir, exist_ok=True)
    rows = []

    for fname in tqdm(files, desc="Evaluating"):
        name = os.path.splitext(fname)[0]
        hr = Image.open(os.path.join(opt.data_dir, fname)).convert("RGB")

        # Crop (not resize) so dimensions are divisible by the scale factor
        w, h = hr.size
        w, h = w - w % s, h - h % s
        hr = hr.crop((0, 0, w, h))

        # Degrade: bicubic downsample; baseline: bicubic upsample back
        lr = hr.resize((w // s, h // s), Image.BICUBIC)
        bic = lr.resize((w, h), Image.BICUBIC)

        # Model inference
        x = torch.from_numpy(np.array(lr)).permute(2, 0, 1).float().div(255.0)
        x = (x * 2.0 - 1.0).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        out = ((out.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().squeeze(0)
        sr_np = out.permute(1, 2, 0).byte().cpu().numpy()

        hr_np = np.array(hr)
        bic_np = np.array(bic)

        b_psnr, b_ssim = y_metrics(bic_np, hr_np, s)
        m_psnr, m_ssim = y_metrics(sr_np, hr_np, s)

        rows.append({
            "image": name,
            "bicubic_psnr": b_psnr, "bicubic_ssim": b_ssim,
            "model_psnr": m_psnr, "model_ssim": m_ssim,
            "gain_psnr": m_psnr - b_psnr,
            "time_s": elapsed,
        })

        if not opt.no_save_images:
            sr_img = Image.fromarray(sr_np)
            sr_img.save(os.path.join(opt.output_dir, f"{name}_sr.png"))
            bic.save(os.path.join(opt.output_dir, f"{name}_bicubic.png"))
            # Panel: LR (enlarged, nearest) | bicubic | SR | HR
            panel = Image.new("RGB", (w * 4, h))
            for i, im in enumerate([lr.resize((w, h), Image.NEAREST), bic, sr_img, hr]):
                panel.paste(im, (i * w, 0))
            panel.save(os.path.join(opt.output_dir, f"{name}_compare.png"))

    # Results table
    print(f"\n{'image':<20}{'bicubic PSNR':>14}{'model PSNR':>12}{'gain':>8}{'bicubic SSIM':>14}{'model SSIM':>12}")
    for r in rows:
        print(f"{r['image']:<20}{r['bicubic_psnr']:>14.2f}{r['model_psnr']:>12.2f}"
              f"{r['gain_psnr']:>+8.2f}{r['bicubic_ssim']:>14.4f}{r['model_ssim']:>12.4f}")

    avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "image"}
    print("-" * 80)
    print(f"{'AVERAGE':<20}{avg['bicubic_psnr']:>14.2f}{avg['model_psnr']:>12.2f}"
          f"{avg['gain_psnr']:>+8.2f}{avg['bicubic_ssim']:>14.4f}{avg['model_ssim']:>12.4f}")
    print(f"Mean inference time: {avg['time_s']:.4f}s per image on {device}")

    csv_path = os.path.join(opt.output_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"image": "AVERAGE", **avg})
    print(f"Saved per-image results to {csv_path}")


if __name__ == "__main__":
    main()
