import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    """Depthwise-separable convolution block used to keep the U-Net small."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvPair(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PicoSAM2BaseUNet(nn.Module):
    """Image-only SMAS segmentation U-Net inspired by PicoSAM2's small U-Net."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 1):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.enc1 = ConvPair(in_channels, c1)
        self.enc2 = ConvPair(c1, c2)
        self.enc3 = ConvPair(c2, c3)
        self.enc4 = ConvPair(c3, c4)
        self.bottleneck = ConvPair(c4, c5)

        self.up4 = ConvPair(c5 + c4, c4)
        self.up3 = ConvPair(c4 + c3, c3)
        self.up2 = ConvPair(c3 + c2, c2)
        self.up1 = ConvPair(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        x = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up4(torch.cat([x, e4], dim=1))
        x = F.interpolate(x, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up3(torch.cat([x, e3], dim=1))
        x = F.interpolate(x, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, e1], dim=1))
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
