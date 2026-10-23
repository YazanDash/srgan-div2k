import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGLoss(nn.Module):
    """
    Perceptual loss: L1 distance between VGG19 feature maps (up to relu5_4)
    of the generated image and the ground-truth image.

    Inputs are expected in [-1, 1] (the range used by the generator and the
    data loader). They are converted to [0, 1] and then ImageNet-normalised
    before being fed to VGG. The VGG weights are frozen.

    Note: the first use downloads the pretrained VGG19 weights (~550 MB).
    """

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()


        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def to_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        # [-1, 1] -> [0, 1] -> ImageNet standard normalisation
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_features = self.vgg(self.to_imagenet(fake))
        real_features = self.vgg(self.to_imagenet(real))
        return self.criterion(fake_features, real_features)
