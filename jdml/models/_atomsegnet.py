"""AtomSegNet denoising architectures.

The model structures in this module are adapted from AtomSegNet:
https://github.com/xinhuolin/AtomSegNet

Copyright (c) 2019 xinhuolin
Licensed under the MIT License. See ``licenses/ATOMSEGNET_LICENSE.txt``.

Only model definitions are included here. Image normalization, padding,
checkpoint loading, and inference are intentionally outside this module.
"""

from __future__ import annotations

import warnings
from functools import partial
from numbers import Integral

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["AtomSegNetUNet", "AtomSegNetNestedUNet"]

_DEFAULT_FEATURES = (32, 64, 128, 256, 512)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _feature_tuple(features: tuple[int, ...]) -> tuple[int, int, int, int, int]:
    if isinstance(features, (str, bytes)):
        raise ValueError("features must contain exactly five positive integers.")
    try:
        values = tuple(features)
    except TypeError as error:
        raise ValueError(
            "features must contain exactly five positive integers."
        ) from error
    if len(values) != 5:
        raise ValueError("features must contain exactly five positive integers.")
    return tuple(
        _positive_integer(f"features[{index}]", value)
        for index, value in enumerate(values)
    )


def _validate_input(
    x: Tensor,
    *,
    in_channels: int,
    downsample_factor: int,
) -> None:
    if x.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor in (N, C, H, W) layout, got {x.ndim}D."
        )
    if x.shape[1] != in_channels:
        raise ValueError(
            f"Expected {in_channels} input channels, got {x.shape[1]}."
        )
    if not x.is_floating_point():
        raise ValueError("Input must use a floating-point dtype.")
    height, width = x.shape[-2:]
    if height < downsample_factor or width < downsample_factor:
        raise ValueError(
            f"Input height and width must each be at least {downsample_factor}."
        )
    if height % downsample_factor or width % downsample_factor:
        raise ValueError(
            "Input height and width must both be divisible by "
            f"{downsample_factor}, got ({height}, {width})."
        )


class AtomSegNetUNet(nn.Module):
    """Shallow U-Net compatible with AtomSegNet's ``denoise.pth``.

    Input tensors have shape ``(N, in_channels, H, W)`` and output tensors have
    shape ``(N, out_channels, H, W)``. Height and width must be divisible by
    four so both skip concatenations have matching spatial dimensions. The
    sigmoid output lies in ``[0, 1]``. The default channel counts preserve the
    original checkpoint architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        *,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        if channels is not None:
            if in_channels != 1 or out_channels != 1:
                raise ValueError(
                    "channels cannot be combined with in_channels or out_channels."
                )
            warnings.warn(
                "channels is deprecated; use in_channels and out_channels instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            in_channels = out_channels = channels

        self.in_channels = _positive_integer("in_channels", in_channels)
        self.out_channels = _positive_integer("out_channels", out_channels)

        self.conv1_1 = nn.Conv2d(self.in_channels, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.upconv4 = nn.Conv2d(256, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn4_out = nn.BatchNorm2d(256)

        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.upconv7 = nn.Conv2d(128, 64, 1)
        self.bn7 = nn.BatchNorm2d(64)
        self.bn7_out = nn.BatchNorm2d(128)

        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.conv9_3 = nn.Conv2d(64, self.out_channels, 1)
        # Retained for strict compatibility with AtomSegNet's state dict even
        # though the original sigmoid architecture does not call this layer.
        self.bn9_3 = nn.BatchNorm2d(self.out_channels)
        self.bn9 = nn.BatchNorm2d(self.out_channels)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        _validate_input(x, in_channels=self.in_channels, downsample_factor=4)
        x1 = F.relu(
            self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x)))))
        )
        x2 = F.relu(
            self.bn2_2(
                self.conv2_2(F.relu(self.bn2_1(self.conv2_1(self.maxpool(x1)))))
            )
        )
        x_up = F.relu(
            self.bn4_2(
                self.conv4_2(F.relu(self.bn4_1(self.conv4_1(self.maxpool(x2)))))
            )
        )
        x_up = self.bn4(self.upconv4(self.upsample(x_up)))
        x_up = self.bn4_out(torch.cat((x2, x_up), dim=1))
        x_up = F.relu(
            self.bn7_2(self.conv7_2(F.relu(self.bn7_1(self.conv7_1(x_up)))))
        )
        x_up = self.bn7(self.upconv7(self.upsample(x_up)))
        x_up = self.bn7_out(torch.cat((x1, x_up), dim=1))
        x_up = F.relu(
            self.conv9_3(
                F.relu(
                    self.bn9_2(
                        self.conv9_2(F.relu(self.bn9_1(self.conv9_1(x_up))))
                    )
                )
            )
        )
        return torch.sigmoid(self.bn9(x_up))

    @classmethod
    def from_config(cls, config: dict) -> "AtomSegNetUNet":
        """Construct a model from a configuration dictionary."""
        return cls(**config)

    def get_config(self) -> dict:
        """Return all constructor arguments needed to reproduce the model."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }


class _VGGBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class AtomSegNetNestedUNet(nn.Module):
    """UNet++ compatible with AtomSegNet's ``Gen1-noNoise.pth``.

    Input tensors have shape ``(N, in_channels, H, W)`` and output tensors have
    shape ``(N, out_channels, H, W)``. Height and width must be divisible by
    16. The tanh output lies in ``[-1, 1]``. The default channel counts and
    features preserve the original checkpoint architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, int, int, int, int] = _DEFAULT_FEATURES,
        *,
        filters: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if filters is not None:
            if features != _DEFAULT_FEATURES:
                raise ValueError("filters cannot be combined with features.")
            warnings.warn(
                "filters is deprecated; use features instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            features = filters

        self.in_channels = _positive_integer("in_channels", in_channels)
        self.out_channels = _positive_integer("out_channels", out_channels)
        self.features_config = _feature_tuple(features)
        features = self.features_config

        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)

        self.conv0_0 = _VGGBlock(self.in_channels, features[0], features[0])
        self.conv1_0 = _VGGBlock(features[0], features[1], features[1])
        self.conv2_0 = _VGGBlock(features[1], features[2], features[2])
        self.conv3_0 = _VGGBlock(features[2], features[3], features[3])
        self.conv4_0 = _VGGBlock(features[3], features[4], features[4])

        self.conv0_1 = _VGGBlock(features[0] + features[1], features[0], features[0])
        self.conv1_1 = _VGGBlock(features[1] + features[2], features[1], features[1])
        self.conv2_1 = _VGGBlock(features[2] + features[3], features[2], features[2])
        self.conv3_1 = _VGGBlock(features[3] + features[4], features[3], features[3])

        self.conv0_2 = _VGGBlock(
            features[0] * 2 + features[1], features[0], features[0]
        )
        self.conv1_2 = _VGGBlock(
            features[1] * 2 + features[2], features[1], features[1]
        )
        self.conv2_2 = _VGGBlock(
            features[2] * 2 + features[3], features[2], features[2]
        )

        self.conv0_3 = _VGGBlock(
            features[0] * 3 + features[1], features[0], features[0]
        )
        self.conv1_3 = _VGGBlock(
            features[1] * 3 + features[2], features[1], features[1]
        )
        self.conv0_4 = _VGGBlock(
            features[0] * 4 + features[1], features[0], features[0]
        )

        self.final = nn.Conv2d(features[0], self.out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        _validate_input(x, in_channels=self.in_channels, downsample_factor=16)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.up(x1_0)), dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.up(x2_0)), dim=1))
        x0_2 = self.conv0_2(torch.cat((x0_0, x0_1, self.up(x1_1)), dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.up(x3_0)), dim=1))
        x1_2 = self.conv1_2(torch.cat((x1_0, x1_1, self.up(x2_1)), dim=1))
        x0_3 = self.conv0_3(
            torch.cat((x0_0, x0_1, x0_2, self.up(x1_2)), dim=1)
        )

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat((x3_0, self.up(x4_0)), dim=1))
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_1, self.up(x3_1)), dim=1))
        x1_3 = self.conv1_3(
            torch.cat((x1_0, x1_1, x1_2, self.up(x2_2)), dim=1)
        )
        x0_4 = self.conv0_4(
            torch.cat((x0_0, x0_1, x0_2, x0_3, self.up(x1_3)), dim=1)
        )
        return torch.tanh(self.final(x0_4))

    @classmethod
    def from_config(cls, config: dict) -> "AtomSegNetNestedUNet":
        """Construct a model from a configuration dictionary."""
        return cls(**config)

    def get_config(self) -> dict:
        """Return all constructor arguments needed to reproduce the model."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "features": self.features_config,
        }
