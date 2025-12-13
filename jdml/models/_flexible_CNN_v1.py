import torch
import torch.nn as nn
from typing import Sequence, Tuple, Optional, List


class CNNModel(nn.Module):
    """
    Generic CNN for image classification.

    Parameters
    ----------
    input_shape : tuple of int
        (C, H, W) of the input images.
    num_classes : int
        Number of output classes.
    conv_arch : sequence of 5-tuples
        Each tuple: (out_channels, kernel_size, stride, padding, pool_kernel)

        Example:
            conv_arch = [
                (32, 3, 1, 1, 2),  # Conv( in→32 ), BN, ReLU, MaxPool(2)
                (64, 3, 1, 1, 2),  # Conv(32→64), BN, ReLU, MaxPool(2)
                (128, 3, 1, 1, 0), # Conv(64→128), BN, ReLU, no pool
            ]
    hidden_dims : sequence of int, optional
        Sizes of hidden fully connected layers before the final classifier.
        Example: [256, 128] gives Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear(num_classes)
    dropout : float, optional
        Dropout probability used in the fully connected part (default 0.0).
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            conv_arch: Sequence[Tuple[int, int, int, int, int]],
            hidden_dims: Optional[Sequence[int]] = None,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_arch = conv_arch
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else []
        self.dropout_p = float(dropout)

        self.features = self._make_conv_layers()
        self.classifier = self._make_classifier()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_conv_layers(self) -> nn.Sequential:
        """
        Build convolutional feature extractor from conv_arch.
        """
        layers: List[nn.Module] = []
        in_channels = self.input_shape[0]

        for i, (out_ch, k, stride, padding, pool_k) in enumerate(self.conv_arch):
            # Conv2d + BatchNorm + ReLU block
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))

            # Optional pooling
            if pool_k is not None and pool_k > 0:
                layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))

            in_channels = out_ch

        return nn.Sequential(*layers)

    def _make_classifier(self) -> nn.Sequential:
        """
        Build fully connected classifier from hidden_dims and feature size.
        """
        # Infer flattened feature size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            feat = self.features(dummy)
            feat_dim = feat.view(1, -1).shape[1]

        dims = [feat_dim] + self.hidden_dims + [self.num_classes]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            # Only add activation + dropout for hidden layers
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if self.dropout_p > 0:
                    layers.append(nn.Dropout(self.dropout_p))

        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: 3×64×64 input, 10 classes
    input_shape = (3, 64, 64)
    num_classes = 10

    # Define conv architecture:
    #  - Conv(3→32) + BN + ReLU + MaxPool(2)
    #  - Conv(32→64) + BN + ReLU + MaxPool(2)
    #  - Conv(64→128) + BN + ReLU (no pool)
    conv_arch = [
        (32, 3, 1, 1, 2),
        (64, 3, 1, 1, 2),
        (128, 3, 1, 1, 0),
    ]

    # Two FC hidden layers: 256, 128
    model = CNNModel(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_arch=conv_arch,
        hidden_dims=[256, 128],
        dropout=0.5,
    )

    # Test forward
    x = torch.randn(8, *input_shape)  # batch of 8 images
    logits = model(x)
    print("Output shape:", logits.shape)  # (8, 10)
