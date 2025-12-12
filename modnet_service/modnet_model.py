"""
Compact MODNet architecture definition.

This file vendors the reference MODNet model (ZHKKKe/MODNet) alongside a
MobileNetV2 backbone so we can load the official portrait-matting
checkpoint without pulling the entire upstream repository at runtime.

Only lightweight adjustments were made:
 - Optional backbone checkpoint loading instead of hard exit.
 - Minor comments for clarity.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# MobileNetV2 backbone (adapted from upstream MODNet)
# ------------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expansion: int, dilation: int = 1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, dilation=dilation, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels: int, alpha: float = 1.0, expansion: int = 6, num_classes: Optional[int] = 1000):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [expansion, 24, 2, 2],
            [expansion, 32, 3, 2],
            [expansion, 64, 4, 2],
            [expansion, 96, 3, 1],
            [expansion, 160, 3, 2],
            [expansion, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * alpha, 8)
        self.last_channel = _make_divisible(last_channel * alpha, 8) if alpha > 1.0 else last_channel
        self.features = [conv_bn(self.in_channels, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(int(c * alpha), 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expansion=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        if self.num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Layout kept identical to upstream for compatibility with checkpoints
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x = self.features[18](x)

        if self.num_classes is not None:
            x = x.mean(dim=(2, 3))
            x = self.classifier(x)

        return x

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_pretrained_model(self, pretrained_file: str) -> None:
        pretrain_dict = torch.load(pretrained_file, map_location="cpu")
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class BaseBackbone(nn.Module):
    """Superclass for plug-and-play backbones."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.model: Optional[nn.Module] = None
        self.enc_channels: List[int] = []

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # pragma: no cover - interface only
        raise NotImplementedError

    def load_pretrained_ckpt(self, checkpoint_path: Optional[str] = None) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """MobileNetV2 backbone yielding multi-scale encoder features."""

    def __init__(self, in_channels: int):
        super().__init__(in_channels)
        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None)
        self.enc_channels = [16, 24, 32, 96, 1280]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x

        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self, checkpoint_path: Optional[str] = None) -> None:
        if not checkpoint_path:
            return
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)


SUPPORTED_BACKBONES = {
    "mobilenetv2": MobileNetV2Backbone,
}


# ------------------------------------------------------------------------------
# MODNet core (adapted from upstream MODNet)
# ------------------------------------------------------------------------------


class IBNorm(nn.Module):
    """Combine Instance Norm and Batch Norm into one layer."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        bn_x = self.bnorm(x[:, : self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels :, ...].contiguous())
        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """Convolution + IBNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_ibn: bool = True,
        with_relu: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.layers(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel reweighting."""

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


class LRBranch(nn.Module):
    """Low-resolution semantic branch."""

    def __init__(self, backbone: BaseBackbone):
        super().__init__()
        enc_channels = backbone.enc_channels
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(
            enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False
        )

    def forward(self, img: torch.Tensor, inference: bool):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode="bilinear", align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode="bilinear", align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """High-resolution detail branch for hair/edges."""

    def __init__(self, hr_channels: int, enc_channels: List[int]):
        super().__init__()
        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img: torch.Tensor, enc2x: torch.Tensor, enc4x: torch.Tensor, lr8x: torch.Tensor, inference: bool):
        img2x = F.interpolate(img, scale_factor=1 / 2, mode="bilinear", align_corners=False)
        img4x = F.interpolate(img, scale_factor=1 / 4, mode="bilinear", align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2, mode="bilinear", align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode="bilinear", align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode="bilinear", align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """Fusion branch that combines semantic and detail cues into the final matte."""

    def __init__(self, hr_channels: int, enc_channels: List[int]):
        super().__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img: torch.Tensor, lr8x: torch.Tensor, hr2x: torch.Tensor) -> torch.Tensor:
        lr4x = F.interpolate(lr8x, scale_factor=2, mode="bilinear", align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode="bilinear", align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode="bilinear", align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)
        return pred_matte


class MODNet(nn.Module):
    """Full MODNet architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        hr_channels: int = 32,
        backbone_arch: str = "mobilenetv2",
        backbone_pretrained: bool = True,
        backbone_ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)
        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            try:
                self.backbone.load_pretrained_ckpt(backbone_ckpt_path)
            except FileNotFoundError:
                # Fallback to randomly initialized backbone when no checkpoint is provided.
                pass

    def forward(self, img: torch.Tensor, inference: bool):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)
        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self) -> None:
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()

    def _init_conv(self, conv: nn.Conv2d) -> None:
        nn.init.kaiming_uniform_(conv.weight, a=0, mode="fan_in", nonlinearity="relu")
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm: nn.modules.batchnorm._NormBase) -> None:
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
