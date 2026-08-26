from collections.abc import Sequence
from numbers import Integral, Real
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    """Return the largest requested-or-smaller divisor of num_channels."""
    num_groups = min(requested_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return num_groups


def _make_activation(name: str) -> nn.Module:
    """Create an activation module from a supported name."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": lambda: nn.ELU(inplace=True),
        "gelu": nn.GELU,
        "silu": lambda: nn.SiLU(inplace=True),
    }
    try:
        return activations[name.lower()]()
    except KeyError as error:
        raise ValueError(
            f"Unknown activation: {name!r}. Choose from {list(activations)}."
        ) from error


def _make_normalization(
    name: Optional[str],
    num_channels: int,
    group_norm_groups: int,
) -> nn.Module:
    """Create a normalization layer for a convolutional feature map."""
    if name is None or name.lower() == "none":
        return nn.Identity()

    name = name.lower()
    if name == "batch":
        return nn.BatchNorm2d(num_channels)
    if name == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if name == "group":
        num_groups = _resolve_group_count(num_channels, group_norm_groups)
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(
        f"Unknown normalization: {name!r}. "
        "Choose from ['batch', 'instance', 'group', 'none']."
    )


def _make_output_activation(name: Optional[str]) -> nn.Module:
    """Create an optional output activation; None leaves outputs as logits."""
    if name is None or name.lower() in {"none", "identity"}:
        return nn.Identity()

    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softmax":
        return nn.Softmax(dim=1)

    raise ValueError(
        f"Unknown output activation: {name!r}. "
        "Choose from ['sigmoid', 'softmax', 'none']."
    )


class DoubleConv(nn.Module):
    """Two same-padded convolutions used throughout the U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        normalization: Optional[str],
        dropout: float,
        group_norm_groups: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        use_bias = normalization is None or normalization.lower() == "none"
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=use_bias,
            ),
            _make_normalization(normalization, out_channels, group_norm_groups),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=use_bias,
            ),
            _make_normalization(normalization, out_channels, group_norm_groups),
            _make_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetModel(nn.Module):
    """Configurable 2D U-Net for segmentation and image-to-image tasks.

    Parameters
    ----------
    in_channels:
        Number of channels in the input image.
    out_channels:
        Number of output channels. For segmentation, this is normally the
        number of classes (or 1 for binary segmentation).
    features:
        Encoder channel widths. Its length controls the depth of the network.
        For example, ``(32, 64, 128, 256)`` creates four encoder levels.
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
        which returns logits and is appropriate for PyTorch's ``*WithLogits``
        and ``CrossEntropyLoss`` functions.
    kernel_size:
        Kernel size for all encoder, bottleneck, and decoder convolutions.
        It must be a positive odd integer so same-padding preserves spatial
        dimensions. Pooling, upsampling, and the final projection keep their
        standard 2x2, 2x2, and 1x1 kernels, respectively.

    Notes
    -----
    Inputs use ``(N, C, H, W)`` layout. Skip features are resized to the
    decoder's current spatial shape, so odd and non-square image sizes are
    supported and the output has the same height and width as the input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        activation: str = "relu",
        normalization: Optional[str] = "batch",
        dropout: float = 0.0,
        up_mode: str = "transpose",
        group_norm_groups: int = 8,
        output_activation: Optional[str] = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        try:
            features = tuple(features)
        except TypeError as error:
            raise ValueError("features must be a sequence of positive integers.") from error

        self._validate_config(
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

        # Normalize numeric scalar types (for example, NumPy integers) to
        # built-in Python types before passing them to PyTorch constructors.
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
        self.dropout_rate = float(dropout)
        self.up_mode = up_mode.lower()
        self.group_norm_groups = group_norm_groups
        self.kernel_size = kernel_size
        self.output_activation_name = (
            None if output_activation is None else output_activation.lower()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()

        current_channels = in_channels
        for feature_channels in features:
            self.encoders.append(
                DoubleConv(
                    current_channels,
                    feature_channels,
                    self.activation_name,
                    self.normalization_name,
                    dropout,
                    group_norm_groups,
                    kernel_size,
                )
            )
            current_channels = feature_channels

        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(
            features[-1],
            bottleneck_channels,
            self.activation_name,
            self.normalization_name,
            dropout,
            group_norm_groups,
            kernel_size,
        )

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_channels

        for feature_channels in reversed(features):
            if self.up_mode == "transpose":
                upsampler = nn.ConvTranspose2d(
                    current_channels,
                    feature_channels,
                    kernel_size=2,
                    stride=2,
                )
            else:
                upsampler = nn.Conv2d(
                    current_channels,
                    feature_channels,
                    kernel_size=1,
                )

            self.upsamplers.append(upsampler)
            self.decoders.append(
                DoubleConv(
                    feature_channels * 2,
                    feature_channels,
                    self.activation_name,
                    self.normalization_name,
                    dropout,
                    group_norm_groups,
                    kernel_size,
                )
            )
            current_channels = feature_channels

        self.output_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1,
        )
        self.output_activation = _make_output_activation(output_activation)

    @staticmethod
    def _validate_config(
        in_channels: int,
        out_channels: int,
        features: tuple[int, ...],
        activation: str,
        normalization: Optional[str],
        dropout: float,
        up_mode: str,
        group_norm_groups: int,
        output_activation: Optional[str],
        kernel_size: int,
    ) -> None:
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, Integral)
            or in_channels <= 0
            or isinstance(out_channels, bool)
            or not isinstance(out_channels, Integral)
            or out_channels <= 0
        ):
            raise ValueError(
                "in_channels and out_channels must be positive integers."
            )
        if not features or any(
            isinstance(width, bool)
            or not isinstance(width, Integral)
            or width <= 0
            for width in features
        ):
            raise ValueError("features must contain at least one positive integer.")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")
        if not isinstance(activation, str):
            raise ValueError("activation must be a supported string name.")
        if normalization is not None and not isinstance(normalization, str):
            raise ValueError(
                "normalization must be a supported string name or None."
            )
        if not isinstance(up_mode, str):
            raise ValueError("up_mode must be 'transpose' or 'bilinear'.")
        if up_mode.lower() not in {"transpose", "bilinear"}:
            raise ValueError("up_mode must be 'transpose' or 'bilinear'.")
        if (
            isinstance(group_norm_groups, bool)
            or not isinstance(group_norm_groups, Integral)
            or group_norm_groups <= 0
        ):
            raise ValueError("group_norm_groups must be a positive integer.")
        if output_activation is not None and not isinstance(output_activation, str):
            raise ValueError(
                "output_activation must be a supported string name or None."
            )
        if (
            isinstance(output_activation, str)
            and output_activation.lower() == "softmax"
            and out_channels < 2
        ):
            raise ValueError(
                "output_activation='softmax' requires out_channels >= 2. "
                "Use logits (None) or 'sigmoid' for a one-channel output."
            )
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, Integral)
            or kernel_size <= 0
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer.")

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
            bottleneck_channels = self.features_config[-1] * 2
            num_groups = _resolve_group_count(
                bottleneck_channels,
                self.group_norm_groups,
            )
            values_per_group = (
                x.shape[0]
                * (bottleneck_channels // num_groups)
                * spatial_elements
            )
            if values_per_group <= 1:
                raise ValueError(
                    "Group normalization needs more than one value per group "
                    "at the bottleneck. Increase the batch size, image size, "
                    "or channels per group, or use normalization='none'."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        skip_connections = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsampler, decoder, skip in zip(
            self.upsamplers,
            self.decoders,
            reversed(skip_connections),
        ):
            if self.up_mode == "bilinear":
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                x = upsampler(x)
            else:
                x = upsampler(x)
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(
                        x,
                        size=skip.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

            x = torch.cat((skip, x), dim=1)
            x = decoder(x)

        x = self.output_conv(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.output_activation(x)

    @classmethod
    def from_config(cls, config: dict) -> "UNetModel":
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
        }
