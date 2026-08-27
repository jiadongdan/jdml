from collections.abc import Sequence
from numbers import Integral, Real

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasFreeLayerNorm(nn.Module):
    """Channel-wise variance normalization without centering or bias."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    """Channel-wise LayerNorm with learnable scale and bias."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    """Apply either Restormer LayerNorm variant over image channels."""

    def __init__(self, dim: int, layer_norm_type: str) -> None:
        super().__init__()
        if layer_norm_type == "bias_free":
            self.body = BiasFreeLayerNorm(dim)
        elif layer_norm_type == "with_bias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError(
                "layer_norm_type must be 'bias_free' or 'with_bias'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        tokens = self.body(tokens)
        return (
            tokens.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class GatedDconvFeedForward(nn.Module):
    """Gated depthwise-convolution feed-forward network (GDFN)."""

    def __init__(
        self,
        dim: int,
        expansion_factor: float,
        bias: bool,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, values = self.depthwise_conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(gate) * values)


class MultiDconvHeadTransposedAttention(nn.Module):
    """Restormer's channel-wise multi-head transposed attention (MDTA)."""

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_depthwise_conv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        channels_per_head = channels // self.num_heads
        q, k, v = self.qkv_depthwise_conv(self.qkv(x)).chunk(3, dim=1)

        shape = (batch, self.num_heads, channels_per_head, height * width)
        q = q.reshape(shape)
        k = k.reshape(shape)
        v = v.reshape(shape)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)

        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project_out(output)


class RestormerBlock(nn.Module):
    """Pre-normalized MDTA and GDFN residual block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim, layer_norm_type)
        self.attention = MultiDconvHeadTransposedAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, layer_norm_type)
        self.feed_forward = GatedDconvFeedForward(
            dim,
            expansion_factor,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class OverlapPatchEmbedding(nn.Module):
    """Overlapping 3x3 projection from image space to feature space."""

    def __init__(self, in_channels: int, dim: int, bias: bool) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class Downsample(nn.Module):
    """Halve spatial resolution and double channels using PixelUnshuffle."""

    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            channels,
            channels // 2,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unshuffle(self.projection(x))


class Upsample(nn.Module):
    """Double spatial resolution and halve channels using PixelShuffle."""

    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.projection(x))


class RestormerModel(nn.Module):
    """Pure-PyTorch Restormer for high-resolution image restoration.

    This implementation follows the architecture described in "Restormer:
    Efficient Transformer for High-Resolution Image Restoration" (CVPR 2022)
    while providing a standalone jdml-style API without BasicSR or einops.

    Parameters
    ----------
    in_channels, out_channels:
        Input and restored-image channel counts.
    dim:
        Base feature width. It must be a positive even integer.
    num_blocks:
        Transformer block counts for encoder levels 1-3 and the latent level.
    num_refinement_blocks:
        Number of full-resolution refinement blocks after decoding.
    heads:
        Attention head counts for levels 1-4.
    ffn_expansion_factor:
        Hidden-width multiplier in each gated feed-forward network.
    bias:
        Whether convolutional projections use bias parameters.
    layer_norm_type:
        ``with_bias`` or ``bias_free``. Official-style names ``WithBias`` and
        ``BiasFree`` are accepted case-insensitively.
    residual:
        Add the input image to the predicted restoration. This requires equal
        input and output channel counts.
    pad_mode:
        PyTorch padding mode used to reach a multiple of 8 before the three
        downsampling stages: ``reflect``, ``replicate``, or ``constant``.
        Reflection automatically falls back to replication for very small
        dimensions where reflection padding is undefined.

    Notes
    -----
    Inputs use ``(N, C, H, W)`` layout. Arbitrary positive H and W are padded
    internally and the output is cropped back to the original image shape.
    By default, the model returns unconstrained restored pixel values.
    """

    downsample_factor = 8

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: Sequence[int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: Sequence[int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "with_bias",
        residual: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()

        try:
            num_blocks = tuple(num_blocks)
            heads = tuple(heads)
        except TypeError as error:
            raise ValueError("num_blocks and heads must be sequences.") from error

        layer_norm_type = self._normalize_layer_norm_type(layer_norm_type)
        self._validate_config(
            in_channels,
            out_channels,
            dim,
            num_blocks,
            num_refinement_blocks,
            heads,
            ffn_expansion_factor,
            bias,
            residual,
            pad_mode,
        )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dim = int(dim)
        self.num_blocks_config = tuple(int(count) for count in num_blocks)
        self.num_refinement_blocks = int(num_refinement_blocks)
        self.heads_config = tuple(int(count) for count in heads)
        self.ffn_expansion_factor = float(ffn_expansion_factor)
        self.bias = bias
        self.layer_norm_type = layer_norm_type
        self.residual = residual
        self.pad_mode = pad_mode.lower()

        level_dims = tuple(self.dim * (2**level) for level in range(4))
        self.patch_embed = OverlapPatchEmbedding(
            self.in_channels,
            level_dims[0],
            bias,
        )

        self.encoder_level1 = self._make_blocks(level_dims[0], 0)
        self.down1_2 = Downsample(level_dims[0], bias)
        self.encoder_level2 = self._make_blocks(level_dims[1], 1)
        self.down2_3 = Downsample(level_dims[1], bias)
        self.encoder_level3 = self._make_blocks(level_dims[2], 2)
        self.down3_4 = Downsample(level_dims[2], bias)
        self.latent = self._make_blocks(level_dims[3], 3)

        self.up4_3 = Upsample(level_dims[3], bias)
        self.reduce_channels_level3 = nn.Conv2d(
            level_dims[3],
            level_dims[2],
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level3 = self._make_blocks(level_dims[2], 2)

        self.up3_2 = Upsample(level_dims[2], bias)
        self.reduce_channels_level2 = nn.Conv2d(
            level_dims[2],
            level_dims[1],
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level2 = self._make_blocks(level_dims[1], 1)

        self.up2_1 = Upsample(level_dims[1], bias)
        decoder_level1_dim = level_dims[0] * 2
        self.decoder_level1 = self._make_block_sequence(
            decoder_level1_dim,
            self.num_blocks_config[0],
            self.heads_config[0],
        )
        self.refinement = self._make_block_sequence(
            decoder_level1_dim,
            self.num_refinement_blocks,
            self.heads_config[0],
        )
        self.output = nn.Conv2d(
            decoder_level1_dim,
            self.out_channels,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

    @staticmethod
    def _normalize_layer_norm_type(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("layer_norm_type must be a supported string name.")
        normalized = name.lower().replace("-", "_")
        aliases = {
            "withbias": "with_bias",
            "with_bias": "with_bias",
            "biasfree": "bias_free",
            "bias_free": "bias_free",
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                "layer_norm_type must be 'with_bias' or 'bias_free'."
            ) from error

    @staticmethod
    def _is_positive_integer(value) -> bool:
        return (
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and value > 0
        )

    @classmethod
    def _validate_config(
        cls,
        in_channels,
        out_channels,
        dim,
        num_blocks,
        num_refinement_blocks,
        heads,
        ffn_expansion_factor,
        bias,
        residual,
        pad_mode,
    ) -> None:
        if not cls._is_positive_integer(in_channels) or not cls._is_positive_integer(
            out_channels
        ):
            raise ValueError(
                "in_channels and out_channels must be positive integers."
            )
        if not cls._is_positive_integer(dim) or dim % 2 != 0:
            raise ValueError("dim must be a positive even integer.")
        if len(num_blocks) != 4 or any(
            not cls._is_positive_integer(count) for count in num_blocks
        ):
            raise ValueError("num_blocks must contain four positive integers.")
        if (
            not isinstance(num_refinement_blocks, Integral)
            or isinstance(num_refinement_blocks, bool)
            or num_refinement_blocks < 0
        ):
            raise ValueError(
                "num_refinement_blocks must be a non-negative integer."
            )
        if len(heads) != 4 or any(
            not cls._is_positive_integer(count) for count in heads
        ):
            raise ValueError("heads must contain four positive integers.")

        for level, (width, num_heads) in enumerate(
            zip((dim, dim * 2, dim * 4, dim * 8), heads),
            start=1,
        ):
            if width % num_heads != 0:
                raise ValueError(
                    f"Feature width {width} at level {level} must be divisible "
                    f"by heads[{level - 1}] ({num_heads})."
                )

        if (
            not isinstance(ffn_expansion_factor, Real)
            or isinstance(ffn_expansion_factor, bool)
            or ffn_expansion_factor <= 0
            or int(dim * ffn_expansion_factor) < 1
        ):
            raise ValueError(
                "ffn_expansion_factor must produce at least one hidden channel."
            )
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean.")
        if not isinstance(residual, bool):
            raise ValueError("residual must be a boolean.")
        if residual and in_channels != out_channels:
            raise ValueError(
                "residual=True requires in_channels == out_channels. "
                "Set residual=False when channel counts differ."
            )
        if not isinstance(pad_mode, str) or pad_mode.lower() not in {
            "reflect",
            "replicate",
            "constant",
        }:
            raise ValueError(
                "pad_mode must be 'reflect', 'replicate', or 'constant'."
            )

    def _make_blocks(self, dim: int, level: int) -> nn.Sequential:
        return self._make_block_sequence(
            dim,
            self.num_blocks_config[level],
            self.heads_config[level],
        )

    def _make_block_sequence(
        self,
        dim: int,
        count: int,
        num_heads: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            *(
                RestormerBlock(
                    dim,
                    num_heads,
                    self.ffn_expansion_factor,
                    self.bias,
                    self.layer_norm_type,
                )
                for _ in range(count)
            )
        )

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_height = (-height) % self.downsample_factor
        pad_width = (-width) % self.downsample_factor
        if pad_height == 0 and pad_width == 0:
            return x

        mode = self.pad_mode
        if mode == "reflect" and (pad_height >= height or pad_width >= width):
            mode = "replicate"
        return F.pad(x, (0, pad_width, 0, pad_height), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor in (N, C, H, W) layout, got {x.ndim}D."
            )
        if x.shape[0] == 0:
            raise ValueError("Input batch must contain at least one image.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        if min(x.shape[-2:]) <= 0:
            raise ValueError("Input height and width must be positive.")
        if not x.is_floating_point():
            raise ValueError("Input images must use a floating-point dtype.")

        original_height, original_width = x.shape[-2:]
        padded_input = self._pad_input(x)

        encoder1 = self.encoder_level1(self.patch_embed(padded_input))
        encoder2 = self.encoder_level2(self.down1_2(encoder1))
        encoder3 = self.encoder_level3(self.down2_3(encoder2))
        latent = self.latent(self.down3_4(encoder3))

        decoder3 = self.up4_3(latent)
        decoder3 = torch.cat((decoder3, encoder3), dim=1)
        decoder3 = self.decoder_level3(
            self.reduce_channels_level3(decoder3)
        )

        decoder2 = self.up3_2(decoder3)
        decoder2 = torch.cat((decoder2, encoder2), dim=1)
        decoder2 = self.decoder_level2(
            self.reduce_channels_level2(decoder2)
        )

        decoder1 = self.up2_1(decoder2)
        decoder1 = torch.cat((decoder1, encoder1), dim=1)
        decoder1 = self.decoder_level1(decoder1)
        output = self.output(self.refinement(decoder1))

        if self.residual:
            output = output + padded_input
        return output[..., :original_height, :original_width]

    @classmethod
    def from_config(cls, config: dict) -> "RestormerModel":
        """Construct a model from a configuration dictionary."""
        return cls(**config)

    def get_config(self) -> dict:
        """Return all constructor arguments needed to reproduce the model."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "dim": self.dim,
            "num_blocks": self.num_blocks_config,
            "num_refinement_blocks": self.num_refinement_blocks,
            "heads": self.heads_config,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "bias": self.bias,
            "layer_norm_type": self.layer_norm_type,
            "residual": self.residual,
            "pad_mode": self.pad_mode,
        }
