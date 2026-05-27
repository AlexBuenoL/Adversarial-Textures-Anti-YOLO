from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, cfg

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two conv layers with GroupNorm and ReLU activations"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=4, num_channels=out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=4, num_channels=out_ch),
        nn.ReLU(inplace=True),
    )

def _down(in_ch: int, out_ch: int) -> tuple[nn.Sequential, nn.MaxPool2d]:
    """Encoder: conv block + 2x max-pool"""
    return _conv_block(in_ch, out_ch), nn.MaxPool2d(2)

def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    """Decoder: 2x bilinear upsample + conv block"""
    return _conv_block(in_ch, out_ch)

class PerturbationUNet(nn.Module):
    def __init__(self, base_channels: int = 8, epsilon: float = 8 / 255.0):
        super().__init__()

        self.epsilon = epsilon
        c = base_channels 

        # encoder
        self.enc1_conv = _conv_block(3, c)           # (3) -> (c)
        self.pool1 = nn.MaxPool2d(2)                 # H/2

        self.enc2_conv = _conv_block(c, c * 2)       # (c) -> (2c)
        self.pool2 = nn.MaxPool2d(2)                 # H/4

        self.enc3_conv = _conv_block(c * 2, c * 4)   # (2c) -> (4c)
        self.pool3 = nn.MaxPool2d(2)                 # H/8

        # bottleneck
        self.bottleneck = _conv_block(c * 4, c * 8)  # (4c) -> (8c)

        # decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _conv_block(c * 8 + c * 4, c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = _conv_block(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _conv_block(c * 2 + c, c)

        # output
        self.head = nn.Conv2d(c, 3, kernel_size=1)  # project to RGB

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # encoder
        s1 = self.enc1_conv(x)               # (B, c, H, W)
        s2 = self.enc2_conv(self.pool1(s1))  # (B, 2c, H/2, W/2)
        s3 = self.enc3_conv(self.pool2(s2))  # (B, 4c, H/4, W/4)

        # bottleneck
        b = self.bottleneck(self.pool3(s3))  # (B, 8c, H/8, W/8)

        # decoder
        d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))  # (B, 4c, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))  # (B, 2c, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B, c, H, W)

        # bound perturbation with tanh
        raw = self.head(d1)
        perturbation = self.epsilon * torch.tanh(raw)
        adv_image = torch.clamp(x + perturbation, 0.0, 1.0)

        return adv_image, perturbation

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_perturbation_net(config: Config = cfg) -> PerturbationUNet:
    """To instantiate the perturbation network"""
    net = PerturbationUNet(
        base_channels=config.unet_base_channels,
        epsilon=config.epsilon,
    )
    print(f"[model] PerturbationUNet | parameters: {net.num_parameters:,}")
    return net