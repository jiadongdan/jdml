"""AtomSegNet denoising architectures.

The model structures in this module are adapted from AtomSegNet:
https://github.com/xinhuolin/AtomSegNet

Copyright (c) 2019 xinhuolin
Licensed under the MIT License. See ``licenses/ATOMSEGNET_LICENSE.txt``.

Only model definitions are included here. Image normalization, padding,
checkpoint loading, and inference are intentionally outside this module.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["AtomSegNetUNet", "AtomSegNetNestedUNet"]


class AtomSegNetUNet(nn.Module):
    """Shallow U-Net compatible with AtomSegNet's ``denoise.pth``.

    Input and output tensors have shape ``(N, channels, H, W)``. Height and
    width must be divisible by four so both skip concatenations have matching
    spatial dimensions. The sigmoid output lies in ``[0, 1]``.
    """

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(channels, 64, 3, padding=1)
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
        self.conv9_3 = nn.Conv2d(64, channels, 1)
        self.bn9_3 = nn.BatchNorm2d(channels)
        self.bn9 = nn.BatchNorm2d(channels)

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
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

    Input and output tensors have shape ``(N, 1, H, W)``. Height and width
    must be divisible by 16. The tanh output lies in ``[-1, 1]``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        self.up = partial(F.interpolate, scale_factor=2)

        self.conv0_0 = _VGGBlock(in_channels, filters[0], filters[0])
        self.conv1_0 = _VGGBlock(filters[0], filters[1], filters[1])
        self.conv2_0 = _VGGBlock(filters[1], filters[2], filters[2])
        self.conv3_0 = _VGGBlock(filters[2], filters[3], filters[3])
        self.conv4_0 = _VGGBlock(filters[3], filters[4], filters[4])

        self.conv0_1 = _VGGBlock(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = _VGGBlock(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = _VGGBlock(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = _VGGBlock(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = _VGGBlock(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = _VGGBlock(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = _VGGBlock(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = _VGGBlock(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = _VGGBlock(filters[1] * 3 + filters[2], filters[1], filters[1])
        self.conv0_4 = _VGGBlock(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
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
