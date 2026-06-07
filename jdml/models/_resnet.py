import torch
import torch.nn as nn


def _make_activation(name):
    activations = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "elu": nn.ELU,
        "selu": nn.SELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]()


class BasicResidualBlock(nn.Module):
    """Two-convolution residual block with optional projection shortcut."""

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = _make_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = _make_activation(activation)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act2(out)
        return out


class ResNetModel(nn.Module):
    """
    Configurable compact ResNet for small multi-channel image classification.

    Args:
        input_dim: tuple (channels, height, width)
        num_classes: number of output classes
        stem_channels: output channels for the initial convolution
        stages: list of stage tuples. Each tuple is
            (out_channels, num_blocks, stride)
            where stride is applied by the first block in the stage.
        fc_layers: optional hidden fully connected layer sizes after pooling/flattening
        use_gap: if True, use global average pooling before the classifier
        dropout: dropout used in optional FC blocks
        block_dropout: spatial dropout inside residual blocks
        activation: relu, leaky_relu, elu, or selu
        stem_kernel_size: initial convolution kernel size
        stem_stride: initial convolution stride
        stem_pool: if True, apply a 2x2 max pool after the stem
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        stem_channels=64,
        stages=None,
        fc_layers=None,
        use_gap=True,
        dropout=0.2,
        block_dropout=0.0,
        activation="relu",
        stem_kernel_size=3,
        stem_stride=1,
        stem_pool=False,
    ):
        super().__init__()

        if stages is None:
            stages = [(64, 2, 1), (128, 2, 2), (256, 2, 2)]
        if fc_layers is None:
            fc_layers = []

        self.input_channels, self.input_h, self.input_w = input_dim
        self.num_classes = num_classes
        self.stem_channels = stem_channels
        self.stages_config = stages
        self.fc_layers_config = fc_layers
        self.use_gap = use_gap
        self.dropout_rate = dropout
        self.block_dropout = block_dropout
        self.activation_name = activation
        self.stem_kernel_size = stem_kernel_size
        self.stem_stride = stem_stride
        self.stem_pool = stem_pool

        if self.input_h <= 0 or self.input_w <= 0 or self.input_channels <= 0:
            raise ValueError(f"Invalid input dimensions: {input_dim}")

        padding = stem_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                stem_channels,
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            _make_activation(activation),
        )
        self.stem_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) if stem_pool else nn.Identity()

        in_channels = stem_channels
        self.stages = nn.ModuleList()
        for stage in stages:
            out_channels, num_blocks, stride = self._parse_stage(stage)
            blocks = []
            blocks.append(
                BasicResidualBlock(
                    in_channels,
                    out_channels,
                    stride=stride,
                    activation=activation,
                    dropout=block_dropout,
                )
            )
            for _ in range(1, num_blocks):
                blocks.append(
                    BasicResidualBlock(
                        out_channels,
                        out_channels,
                        stride=1,
                        activation=activation,
                        dropout=block_dropout,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_channels

        final_h, final_w = self._calculate_feature_dims()
        if final_h <= 0 or final_w <= 0:
            raise ValueError(
                f"Feature map dimensions became invalid ({final_h}x{final_w}). "
                "Try reducing strides or disabling stem_pool."
            )

        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            in_features = in_channels
        else:
            self.gap = None
            in_features = in_channels * final_h * final_w

        fc_blocks = []
        for hidden_size in fc_layers:
            fc_blocks.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    _make_activation(activation),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            in_features = hidden_size
        self.fc = nn.Sequential(*fc_blocks) if fc_blocks else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    @staticmethod
    def _parse_stage(stage):
        if isinstance(stage, dict):
            return stage["out_channels"], stage.get("num_blocks", 2), stage.get("stride", 1)
        if len(stage) == 2:
            out_channels, num_blocks = stage
            return out_channels, num_blocks, 1
        if len(stage) == 3:
            return stage
        raise ValueError(
            "Each stage must be (out_channels, num_blocks), "
            "(out_channels, num_blocks, stride), or a dict."
        )

    def _calculate_feature_dims(self):
        h, w = self.input_h, self.input_w
        h = (h + 2 * (self.stem_kernel_size // 2) - self.stem_kernel_size) // self.stem_stride + 1
        w = (w + 2 * (self.stem_kernel_size // 2) - self.stem_kernel_size) // self.stem_stride + 1

        if self.stem_pool:
            h //= 2
            w //= 2

        for stage in self.stages_config:
            _, _, stride = self._parse_stage(stage)
            h = (h + 2 - 3) // stride + 1
            w = (w + 2 - 3) // stride + 1

        return h, w

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_pool_layer(x)

        for stage in self.stages:
            x = stage(x)

        if self.use_gap:
            x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def summary(self, input_size=None):
        if input_size is None:
            input_size = (self.input_channels, self.input_h, self.input_w)

        print("=" * 80)
        print("ResNet Model Architecture")
        print("=" * 80)
        print(f"Input: {input_size}")
        print(f"Output Classes: {self.num_classes}")
        print(f"Stem: {self.input_channels} -> {self.stem_channels}, kernel={self.stem_kernel_size}, stride={self.stem_stride}")
        if self.stem_pool:
            print("Stem Pool: max, kernel=2, stride=2")
        print("-" * 80)

        in_channels = self.stem_channels
        h, w = self._calculate_stem_dims()
        for i, stage in enumerate(self.stages_config, start=1):
            out_channels, num_blocks, stride = self._parse_stage(stage)
            h = (h + 2 - 3) // stride + 1
            w = (w + 2 - 3) // stride + 1
            print(
                f"Stage{i}: {in_channels} -> {out_channels}, "
                f"blocks={num_blocks}, first_stride={stride}, output=({out_channels}, {h}, {w})"
            )
            in_channels = out_channels

        if self.use_gap:
            print(f"Global Average Pooling: ({in_channels}, {h}, {w}) -> ({in_channels}, 1, 1)")
            head_features = in_channels
        else:
            head_features = in_channels * h * w
            print(f"Flatten: ({in_channels}, {h}, {w}) -> {head_features}")

        if self.fc_layers_config:
            print("Fully Connected Layers:")
            in_features = head_features
            for i, hidden_size in enumerate(self.fc_layers_config, start=1):
                print(f"  FC{i}: {in_features} -> {hidden_size} (dropout={self.dropout_rate})")
                in_features = hidden_size
            print(f"  Classifier: {in_features} -> {self.num_classes}")
        else:
            print(f"Classifier: {head_features} -> {self.num_classes}")

        print("-" * 80)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 80)

    def _calculate_stem_dims(self):
        h = (self.input_h + 2 * (self.stem_kernel_size // 2) - self.stem_kernel_size) // self.stem_stride + 1
        w = (self.input_w + 2 * (self.stem_kernel_size // 2) - self.stem_kernel_size) // self.stem_stride + 1
        if self.stem_pool:
            h //= 2
            w //= 2
        return h, w

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "input_dim": (self.input_channels, self.input_h, self.input_w),
            "num_classes": self.num_classes,
            "stem_channels": self.stem_channels,
            "stages": self.stages_config,
            "fc_layers": self.fc_layers_config,
            "use_gap": self.use_gap,
            "dropout": self.dropout_rate,
            "block_dropout": self.block_dropout,
            "activation": self.activation_name,
            "stem_kernel_size": self.stem_kernel_size,
            "stem_stride": self.stem_stride,
            "stem_pool": self.stem_pool,
        }
