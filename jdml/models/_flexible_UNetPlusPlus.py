from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._flexible_UNet import (
    DoubleConv,
    UNetModel,
    _make_activation,
    _make_normalization,
    _make_output_activation,
    _resolve_group_count,
)


class UNetPlusPlusModel(nn.Module):
    """Configurable 2D U-Net++ with nested dense skip connections.

    Parameters
    ----------
    in_channels:
        Number of channels in the input image.
    out_channels:
        Number of output channels. For segmentation, this is normally the
        number of classes (or 1 for binary segmentation).
    features:
        Encoder channel widths. Its length controls both encoder depth and the
        number of nested decoder stages.
    activation:
        One of ``relu``, ``leaky_relu``, ``elu``, ``gelu``, or ``silu``.
    normalization:
        One of ``batch``, ``instance``, ``group``, or ``none``.
    dropout:
        Spatial dropout probability within each double-convolution block.
    up_mode:
        ``transpose`` uses learned transposed convolutions. ``bilinear`` uses
        bilinear interpolation followed by a 1x1 channel projection.
    group_norm_groups:
        Preferred number of groups for group normalization. If a channel width
        is not divisible by this value, the largest smaller divisor is used.
    output_activation:
        Optional ``sigmoid`` or channel-wise ``softmax``. The default is None,
        which returns logits.
    kernel_size:
        Positive odd kernel size for encoder, bottleneck, and nested decoder
        convolutions. Same-padding preserves spatial dimensions.
    deep_supervision:
        If False, ``forward`` returns the final prediction tensor. If True, it
        returns a tuple containing one full-resolution prediction from every
        top-row nested decoder stage, ordered from shallowest to deepest.

    Notes
    -----
    Inputs use ``(N, C, H, W)`` layout. Height and width may be odd, unequal,
    and need not be divisible by the total downsampling factor. Each dimension
    must be at least ``2 ** len(features)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
        activation: str = "relu",
        normalization: Optional[str] = "batch",
        dropout: float = 0.0,
        up_mode: str = "transpose",
        group_norm_groups: int = 8,
        output_activation: Optional[str] = None,
        kernel_size: int = 3,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        try:
            features = tuple(features)
        except TypeError as error:
            raise ValueError(
                "features must be a sequence of positive integers."
            ) from error

        # Reuse the common U-Net configuration contract so both models reject
        # the same invalid channel, activation, normalization, and output
        # configurations.
        UNetModel._validate_config(
            in_channels,
            out_channels,
            features,
            activation,
            normalization,
            dropout,
            up_mode,
            group_norm_groups,
            output_activation,
            kernel_size,
        )
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be a boolean.")

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        features = tuple(int(width) for width in features)
        dropout = float(dropout)
        group_norm_groups = int(group_norm_groups)
        kernel_size = int(kernel_size)

        # Validate names before partially constructing the model.
        _make_activation(activation)
        _make_normalization(normalization, features[0], group_norm_groups)
        _make_output_activation(output_activation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_config = features
        self.activation_name = activation.lower()
        self.normalization_name = (
            "none" if normalization is None else normalization.lower()
        )
        self.dropout_rate = dropout
        self.up_mode = up_mode.lower()
        self.group_norm_groups = group_norm_groups
        self.output_activation_name = (
            None if output_activation is None else output_activation.lower()
        )
        self.kernel_size = kernel_size
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()

        current_channels = in_channels
        for feature_channels in features:
            self.encoders.append(
                self._make_conv_block(current_channels, feature_channels)
            )
            current_channels = feature_channels

        self.bottleneck_channels = features[-1] * 2
        self.bottleneck = self._make_conv_block(
            features[-1],
            self.bottleneck_channels,
        )

        # Grid notation follows the U-Net++ paper: X[i, 0] are encoder nodes,
        # and X[i, j] are nested decoder nodes. Each X[i, j] concatenates all
        # earlier nodes in its row with upsampled X[i + 1, j - 1].
        grid_channels = features + (self.bottleneck_channels,)
        self.upsamplers = nn.ModuleDict()
        self.nested_blocks = nn.ModuleDict()
        depth = len(features)

        for stage in range(1, depth + 1):
            for level in range(depth - stage + 1):
                key = self._node_key(level, stage)
                deeper_channels = grid_channels[level + 1]
                target_channels = grid_channels[level]

                if self.up_mode == "transpose":
                    self.upsamplers[key] = nn.ConvTranspose2d(
                        deeper_channels,
                        target_channels,
                        kernel_size=2,
                        stride=2,
                    )
                else:
                    self.upsamplers[key] = nn.Conv2d(
                        deeper_channels,
                        target_channels,
                        kernel_size=1,
                    )

                concatenated_channels = (stage + 1) * target_channels
                self.nested_blocks[key] = self._make_conv_block(
                    concatenated_channels,
                    target_channels,
                )

        number_of_heads = depth if deep_supervision else 1
        self.output_heads = nn.ModuleList(
            nn.Conv2d(features[0], out_channels, kernel_size=1)
            for _ in range(number_of_heads)
        )
        self.output_activation = _make_output_activation(output_activation)

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> DoubleConv:
        return DoubleConv(
            in_channels,
            out_channels,
            self.activation_name,
            self.normalization_name,
            self.dropout_rate,
            self.group_norm_groups,
            self.kernel_size,
        )

    @staticmethod
    def _node_key(level: int, stage: int) -> str:
        return f"x_{level}_{stage}"

    def _validate_normalization_shape(self, x: torch.Tensor) -> None:
        """Reject bottleneck shapes unsupported by the selected normalization."""
        downsample_factor = 2 ** len(self.features_config)
        bottleneck_h = x.shape[-2] // downsample_factor
        bottleneck_w = x.shape[-1] // downsample_factor
        spatial_elements = bottleneck_h * bottleneck_w

        if (
            self.normalization_name == "batch"
            and self.training
            and x.shape[0] * spatial_elements <= 1
        ):
            raise ValueError(
                "Batch normalization needs more than one value per channel at "
                "the bottleneck during training. Increase the batch size or "
                "image size, or use normalization='group' or 'none'."
            )

        if self.normalization_name == "instance" and spatial_elements <= 1:
            raise ValueError(
                "Instance normalization needs more than one spatial value at "
                "the bottleneck. Increase the image size or use a different "
                "normalization mode."
            )

        if self.normalization_name == "group":
            num_groups = _resolve_group_count(
                self.bottleneck_channels,
                self.group_norm_groups,
            )
            values_per_group = (
                x.shape[0]
                * (self.bottleneck_channels // num_groups)
                * spatial_elements
            )
            if values_per_group <= 1:
                raise ValueError(
                    "Group normalization needs more than one value per group "
                    "at the bottleneck. Increase the batch size, image size, "
                    "or channels per group, or use normalization='none'."
                )

    def _upsample_to(
        self,
        x: torch.Tensor,
        target_size: torch.Size,
        key: str,
    ) -> torch.Tensor:
        if self.up_mode == "bilinear":
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            return self.upsamplers[key](x)

        x = self.upsamplers[key](x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _apply_output_head(
        self,
        x: torch.Tensor,
        head: nn.Module,
        input_size: torch.Size,
    ) -> torch.Tensor:
        x = head(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.output_activation(x)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor in (N, C, H, W) layout, got {x.ndim}D."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        minimum_size = 2 ** len(self.features_config)
        if min(x.shape[-2:]) < minimum_size:
            raise ValueError(
                f"Input height and width must each be at least {minimum_size} "
                f"for {len(self.features_config)} encoder levels."
            )
        self._validate_normalization_shape(x)

        input_size = x.shape[-2:]
        nodes = {}

        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            nodes[(level, 0)] = x
            x = self.pool(x)
        nodes[(len(self.features_config), 0)] = self.bottleneck(x)

        depth = len(self.features_config)
        for stage in range(1, depth + 1):
            for level in range(depth - stage + 1):
                key = self._node_key(level, stage)
                target_size = nodes[(level, 0)].shape[-2:]
                upsampled = self._upsample_to(
                    nodes[(level + 1, stage - 1)],
                    target_size,
                    key,
                )
                dense_inputs = [
                    nodes[(level, previous_stage)]
                    for previous_stage in range(stage)
                ]
                dense_inputs.append(upsampled)
                nodes[(level, stage)] = self.nested_blocks[key](
                    torch.cat(dense_inputs, dim=1)
                )

        if self.deep_supervision:
            return tuple(
                self._apply_output_head(
                    nodes[(0, stage)],
                    head,
                    input_size,
                )
                for stage, head in enumerate(self.output_heads, start=1)
            )

        return self._apply_output_head(
            nodes[(0, depth)],
            self.output_heads[0],
            input_size,
        )

    @classmethod
    def from_config(cls, config: dict) -> "UNetPlusPlusModel":
        """Construct a model from a configuration dictionary."""
        return cls(**config)

    def get_config(self) -> dict:
        """Return all constructor arguments needed to reproduce the model."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "features": self.features_config,
            "activation": self.activation_name,
            "normalization": self.normalization_name,
            "dropout": self.dropout_rate,
            "up_mode": self.up_mode,
            "group_norm_groups": self.group_norm_groups,
            "output_activation": self.output_activation_name,
            "kernel_size": self.kernel_size,
            "deep_supervision": self.deep_supervision,
        }
