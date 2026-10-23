import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as calculate_ssim
import torchvision.transforms.functional as TF

from load import DIV2KDataset, DIV2KValDataset
from loss import VGGLoss
from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Train SRGAN 4x Model')
parser.add_argument('--scale_factor', default=4, type=int, choices=[2, 4, 8], help='Super-resolution upscale factor')
parser.add_argument('--train_crop_size', default=128, type=int, help='Training HR crop size')
parser.add_argument('--val_crop_size', default=256, type=int, help='Validation HR crop size')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
parser.add_argument('--gan_weight', type=float, default=1e-3, help='Weight for GAN loss (set 0.0 for Phase 1)')
parser.add_argument('--load_checkpoint', type=str, default='', help='Path to pre-trained generator for Phase 2')
parser.add_argument('--vgg_weight', type=float, default=0.006, help='0.0 disables the VGG loss (much faster)')
parser.add_argument('--num_res_blocks', type=int, default=8, help='Residual blocks in the generator')
parser.add_argument('--data_root', type=str, default='.', help='Folder that contains the extracted DIV2K_* folders')

def rgb_to_y(img):
    """img: [B, 3, H, W] in [0, 1]. Returns Y channel in [16, 235] (BT.601, same as MATLAB rgb2ycbcr)."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 16.0 + 65.481 * r + 128.553 * g + 24.966 * b


def psnr_y(pred_01, target_01, border=4):
    #PSNR on the Y channel with `border` pixels cropped from every side.
    pred_y = rgb_to_y(pred_01)
    target_y = rgb_to_y(target_01)
    if border > 0:
        pred_y = pred_y[:, :, border:-border, border:-border]
        target_y = target_y[:, :, border:-border, border:-border]
    mse = torch.mean((pred_y - target_y) ** 2)
    return 10 * torch.log10(255.0 ** 2 / (mse + 1e-10))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    opt = parser.parse_args()

    # Create directories for outputs and weights
    os.makedirs("epochs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Datasets & Loaders

    train_hr = os.path.join(opt.data_root, 'DIV2K_train_HR', 'DIV2K_train_HR')
    train_lr = os.path.join(opt.data_root, 'DIV2K_train_LR_bicubic_X4', 'X4')
    valid_hr = os.path.join(opt.data_root, 'DIV2K_valid_HR', 'DIV2K_valid_HR')
    valid_lr = os.path.join(opt.data_root, 'DIV2K_valid_LR_bicubic_X4', 'X4')


    train_set = DIV2KDataset(
        hr_dir=train_hr,
        lr_dir=train_lr,
        hr_crop_size=opt.train_crop_size,
        scale_factor=opt.scale_factor
    )
    val_set = DIV2KValDataset(
        hr_dir=valid_hr,
        lr_dir=valid_lr,
        hr_crop_size=opt.val_crop_size,
        scale_factor=opt.scale_factor
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Networks
    netG = Generator(opt.scale_factor,num_res_blocks=opt.num_res_blocks).to(device)
    netD = Discriminator().to(device)

    if opt.load_checkpoint:
        netG.load_state_dict(torch.load(opt.load_checkpoint, map_location=device))
        print(f"Loaded pre-trained Generator weights from {opt.load_checkpoint}")

    print(f"Generator parameters    : {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    # 3. Optimizers
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=opt.epochs, eta_min=1e-6)

    # 4. Loss Functions
    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = nn.L1Loss().to(device)
    criterion_vgg = VGGLoss().to(device) if opt.vgg_weight > 0 else None

    results = {'d_loss': [], 'g_loss': [], 'psnr': [], 'ssim': []}

    # 5. Training & Validation Loop
    for epoch in range(1, opt.epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0.0, 'g_loss': 0.0}

        netG.train()
        netD.train()

        for lr_img, hr_img in train_bar:

            batch_size = lr_img.size(0)
            running_results['batch_sizes'] += batch_size

            lr_img = lr_img.to(device,non_blocking=True)
            hr_img = hr_img.to(device,non_blocking=True)

            # Generate fake image first (needed for both D and G)
            fake_img = netG(lr_img)


            # Update Discriminator

            if opt.gan_weight > 0.0:
                optimizerD.zero_grad()

                real_out = netD(hr_img)
                d_loss_real = criterion_gan(real_out, torch.ones_like(real_out))

                fake_out = netD(fake_img.detach())
                d_loss_fake = criterion_gan(fake_out, torch.zeros_like(fake_out))

                d_loss = (d_loss_real + d_loss_fake) / 2.0
                d_loss.backward()
                optimizerD.step()
            else:
                d_loss = torch.tensor(0.0, device=device)


            # Update Generator
            optimizerG.zero_grad()

            g_pixel_loss = criterion_pixel(fake_img, hr_img)
            g_vgg_loss = criterion_vgg(fake_img, hr_img) if criterion_vgg is not None else torch.tensor(0.0,
                                                                                                        device=device)

            if opt.gan_weight > 0.0:
                fake_out_for_g = netD(fake_img)
                g_gan_loss = criterion_gan(fake_out_for_g, torch.ones_like(fake_out_for_g))
                g_loss = g_pixel_loss + (opt.vgg_weight * g_vgg_loss) + (opt.gan_weight * g_gan_loss)
            else:
                g_loss = g_pixel_loss + (opt.vgg_weight * g_vgg_loss)

            g_loss.backward()
            optimizerG.step()

            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['g_loss'] += g_loss.item() * batch_size

            train_bar.set_description(
                desc=f"[{epoch}/{opt.epochs}] Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}"
            )

        # Validation

        netG.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        bicubic_psnr = 0.0
        val_count = 0

        with torch.no_grad():
            for val_lr, val_hr in val_loader:
                val_lr = val_lr.to(device)
                val_hr = val_hr.to(device)

                val_fake = netG(val_lr)

                # Un-normalize from [-1, 1] -> [0, 1]
                fake_01 = torch.clamp((val_fake + 1.0) / 2.0, 0.0, 1.0)
                hr_01 = torch.clamp((val_hr + 1.0) / 2.0, 0.0, 1.0)

                # Bicubic baseline
                bic = F.interpolate(val_lr, scale_factor=opt.scale_factor, mode="bicubic", align_corners=False)
                bic_01 = torch.clamp((bic + 1.0) / 2.0, 0.0, 1.0)
                bicubic_psnr += psnr_y(bic_01, hr_01, border=opt.scale_factor).item()

                val_psnr += psnr_y(fake_01, hr_01, border=opt.scale_factor).item()
                fake_np = fake_01.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                hr_np = hr_01.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                val_ssim += calculate_ssim(fake_np, hr_np, channel_axis=2, data_range=1.0)

                val_count += 1

        avg_psnr = val_psnr / val_count
        avg_ssim = val_ssim / val_count
        avg_bicubic = bicubic_psnr / val_count

        print(
            f"Epoch [{epoch}/{opt.epochs}] - Val PSNR: {avg_psnr:.2f} dB | "
            f"Bicubic PSNR: {avg_bicubic:.2f} dB | Gain: {avg_psnr - avg_bicubic:+.2f} dB | "
            f"Val SSIM: {avg_ssim:.4f}"
        )


        sample_img = TF.to_pil_image(fake_01.squeeze(0).cpu())
        sample_img.save(f"epochs/epoch_{epoch}_psnr_{avg_psnr:.2f}.png")

        # Save Model Checkpoint
        torch.save(netG.state_dict(), f"checkpoints/netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"checkpoints/netD_epoch_{epoch}.pth")
        scheduler.step()
