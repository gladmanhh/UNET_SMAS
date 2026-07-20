import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int, groups: int = 1) -> nn.Sequential:
    """Conv followed by BatchNorm, with no bias (BN absorbs it)."""
    block = nn.Sequential()
    block.add_module(
        "conv",
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    )
    block.add_module("bn", nn.BatchNorm2d(out_ch))
    return block


class RepVGGBlock(nn.Module):
    """RepVGG building block.

    Train time: 3x3 conv-bn + 1x1 conv-bn + (optional) identity BN, summed then ReLU.
    Inference time (after ``switch_to_deploy``): a single fused 3x3 conv + ReLU.
    Reparameterization follows `RepVGG: Making VGG-style ConvNets Great Again`
    (https://arxiv.org/abs/2101.03697).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = stride
        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=True
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(in_ch) if out_ch == in_ch and stride == 1 else None
            )
            self.rbr_dense = conv_bn(in_ch, out_ch, 3, stride, 1, groups=groups)
            self.rbr_1x1 = conv_bn(in_ch, out_ch, 1, stride, 0, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(x))
        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + identity)

    # --- reparameterization ---------------------------------------------------
    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn_tensor(self.rbr_dense)
        k1, b1 = self._fuse_bn_tensor(self.rbr_1x1)
        kid, bid = self._fuse_bn_tensor(self.rbr_identity)
        return k3 + self._pad_1x1_to_3x3_tensor(k1) + kid, b3 + b1 + bid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel):
        if isinstance(kernel, int):
            return 0
        return F.pad(kernel, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self) -> None:
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for attr in ("rbr_dense", "rbr_1x1", "rbr_identity", "id_tensor"):
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.deploy = True


class RepVGGPair(nn.Module):
    """Two stacked RepVGG blocks, replacing the depthwise-separable ConvPair."""

    def __init__(self, in_ch: int, out_ch: int, deploy: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            RepVGGBlock(in_ch, out_ch, stride=1, deploy=deploy),
            RepVGGBlock(out_ch, out_ch, stride=1, deploy=deploy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PicoSAM2BaseUNet(nn.Module):
    """Image-only 4-class layer segmentation U-Net built from RepVGG blocks."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 4, deploy: bool = False):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.enc1 = RepVGGPair(in_channels, c1, deploy=deploy)
        self.enc2 = RepVGGPair(c1, c2, deploy=deploy)
        self.enc3 = RepVGGPair(c2, c3, deploy=deploy)
        self.enc4 = RepVGGPair(c3, c4, deploy=deploy)
        self.bottleneck = RepVGGPair(c4, c5, deploy=deploy)

        self.up4 = RepVGGPair(c5 + c4, c4, deploy=deploy)
        self.up3 = RepVGGPair(c4 + c3, c3, deploy=deploy)
        self.up2 = RepVGGPair(c3 + c2, c2, deploy=deploy)
        self.up1 = RepVGGPair(c2 + c1, c1, deploy=deploy)
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


def reparameterize_model(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Fuse every RepVGG block into a single 3x3 conv for fast inference.

    Returns a model in deploy mode. By default operates on a deep copy so the
    original multi-branch model is left intact.
    """
    if not inplace:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    return model


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
