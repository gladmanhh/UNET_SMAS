import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileOneBlock(nn.Module):
    """MobileOne building block.

    Train time: a multi-branched structure (``num_conv_branches`` k x k conv-bn
    branches + an optional 1x1 scale branch + an optional BN skip branch), summed
    then ReLU. Inference time (after ``reparameterize``): a single fused conv + ReLU.
    Paper: `An Improved One millisecond Mobile Backbone`
    (https://arxiv.org/abs/2206.04040).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        inference_mode: bool = False,
        num_conv_branches: int = 1,
    ):
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = nn.ReLU(inplace=True)

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=padding, groups=groups, bias=True,
            )
        else:
            # Re-parameterizable skip connection (only when shape is preserved).
            self.rbr_skip = (
                nn.BatchNorm2d(in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            # Over-parameterized k x k conv-bn branches.
            self.rbr_conv = nn.ModuleList(
                [self._conv_bn(kernel_size, padding) for _ in range(num_conv_branches)]
            )
            # 1x1 scale branch (only meaningful for k > 1).
            self.rbr_scale = self._conv_bn(1, 0) if kernel_size > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self.activation(self.reparam_conv(x))
        identity_out = 0 if self.rbr_skip is None else self.rbr_skip(x)
        scale_out = 0 if self.rbr_scale is None else self.rbr_scale(x)
        out = scale_out + identity_out
        for branch in self.rbr_conv:
            out = out + branch(x)
        return self.activation(out)

    # --- reparameterization ---------------------------------------------------
    def reparameterize(self) -> None:
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=self.kernel_size // 2,
            groups=self.groups, bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for attr in ("rbr_conv", "rbr_scale", "rbr_skip", "id_tensor"):
            if hasattr(self, attr):
                self.__delattr__(attr)
        self.inference_mode = True

    def _get_kernel_bias(self):
        kernel_scale, bias_scale = 0, 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad 1x1 scale kernel out to k x k so branches can be summed.
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity, bias_identity = 0, 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv, bias_conv = 0, 0
        for branch in self.rbr_conv:
            k, b = self._fuse_bn_tensor(branch)
            kernel_conv = kernel_conv + k
            bias_conv = bias_conv + b

        return (
            kernel_conv + kernel_scale + kernel_identity,
            bias_conv + bias_scale + bias_identity,
        )

    def _fuse_bn_tensor(self, branch):
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
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
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

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        mod = nn.Sequential()
        mod.add_module(
            "conv",
            nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size,
                stride=self.stride, padding=padding, groups=self.groups, bias=False,
            ),
        )
        mod.add_module("bn", nn.BatchNorm2d(self.out_channels))
        return mod


class DSConv(nn.Module):
    """Depthwise-separable conv, MobileOne style: depthwise 3x3 block + pointwise 1x1 block.

    Each of the two blocks is a reparameterizable MobileOneBlock, so training uses
    multi-branch blocks while inference collapses to plain depthwise + pointwise convs.
    """

    def __init__(self, in_ch: int, out_ch: int, inference_mode: bool = False, num_conv_branches: int = 1):
        super().__init__()
        self.dw = MobileOneBlock(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch,
            inference_mode=inference_mode, num_conv_branches=num_conv_branches,
        )
        self.pw = MobileOneBlock(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1,
            inference_mode=inference_mode, num_conv_branches=num_conv_branches,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ConvPair(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, inference_mode: bool = False, num_conv_branches: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(in_ch, out_ch, inference_mode, num_conv_branches),
            DSConv(out_ch, out_ch, inference_mode, num_conv_branches),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PicoSAM2BaseUNet(nn.Module):
    """Image-only 4-class layer segmentation U-Net built from MobileOne blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        out_channels: int = 4,
        inference_mode: bool = False,
        num_conv_branches: int = 1,
    ):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        def pair(in_ch, out_ch):
            return ConvPair(in_ch, out_ch, inference_mode=inference_mode, num_conv_branches=num_conv_branches)

        self.enc1 = pair(in_channels, c1)
        self.enc2 = pair(c1, c2)
        self.enc3 = pair(c2, c3)
        self.enc4 = pair(c3, c4)
        self.bottleneck = pair(c4, c5)

        self.up4 = pair(c5 + c4, c4)
        self.up3 = pair(c4 + c3, c3)
        self.up2 = pair(c3 + c2, c2)
        self.up1 = pair(c2 + c1, c1)
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
    """Fuse every MobileOne block into a single conv for fast inference.

    Returns a model in inference (deploy) mode. By default operates on a deep copy
    so the original multi-branch model is left intact.
    """
    if not inplace:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
