import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    Flexible CNN model constructor.

    Args:
        input_dim: tuple (channels, height, width) - input image dimensions
        num_classes: int - number of output classes
        conv_layers: list of tuples defining convolutional layers
            Each tuple: (out_channels, kernel_size, stride, padding)
            Example: [(32, 3, 1, 1), (64, 3, 1, 1)]
        fc_layers: list of ints defining fully connected layer sizes
            Example: [512, 256] creates two FC layers
        pool: tuple (type, kernel_size, stride) - pooling after each conv
            type: 'max' or 'avg'
            Example: ('max', 2, 2)
        dropout: float - dropout rate (0 to disable)
        activation: str - 'relu', 'leaky_relu', 'elu', 'selu'

    Example usage:
        model = CNNModel(
            input_dim=(3, 32, 32),
            num_classes=10,
            conv_layers=[(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)],
            fc_layers=[512, 256],
            pool=('max', 2, 2),
            dropout=0.5
        )
    """
    def __init__(self, input_dim, num_classes, conv_layers, fc_layers=None,
                 pool=('max', 2, 2), dropout=0.5, activation='relu'):
        super(CNNModel, self).__init__()

        self.input_channels, self.input_h, self.input_w = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Build activation function
        self.activation = self._get_activation(activation)

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_channels

        for out_channels, kernel_size, stride, padding in conv_layers:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                self._get_activation(activation)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Build pooling layer
        if pool[0] == 'max':
            self.pool = nn.MaxPool2d(pool[1], pool[2])
        elif pool[0] == 'avg':
            self.pool = nn.AvgPool2d(pool[1], pool[2])
        else:
            raise ValueError(f"Unknown pooling type: {pool[0]}")

        # Calculate feature map size after conv and pooling
        self.flat_features = self._calculate_flat_features(conv_layers, pool)

        # Build fully connected layers
        self.fc_blocks = nn.ModuleList()

        if fc_layers is None:
            fc_layers = []

        in_features = self.flat_features
        for hidden_size in fc_layers:
            block = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.fc_blocks.append(block)
            in_features = hidden_size

        # Final classification layer
        self.classifier = nn.Linear(in_features, num_classes)

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name, nn.ReLU())

    def _calculate_flat_features(self, conv_layers, pool):
        """Calculate the flattened feature size after conv and pool layers"""
        h, w = self.input_h, self.input_w

        for _, _, stride, padding in conv_layers:
            # After pooling
            h = (h + 2 * padding) // stride
            w = (w + 2 * padding) // stride
            h = h // pool[2]
            w = w // pool[2]

        return conv_layers[-1][0] * h * w  # channels * height * width

    def forward(self, x):
        # Convolutional blocks with pooling
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected blocks
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # Classification layer
        x = self.classifier(x)
        return x


# Example 1: Simple CNN for CIFAR-10
model1 = CNNModel(
    input_dim=(3, 32, 32),
    num_classes=10,
    conv_layers=[(32, 3, 1, 1), (64, 3, 1, 1)],
    fc_layers=[512],
    pool=('max', 2, 2),
    dropout=0.5
)

# Example 2: Deeper CNN for ImageNet-style
model2 = CNNModel(
    input_dim=(3, 224, 224),
    num_classes=1000,
    conv_layers=[(64, 3, 1, 1), (128, 3, 1, 1), (256, 3, 1, 1), (512, 3, 1, 1)],
    fc_layers=[2048, 1024],
    pool=('max', 2, 2),
    dropout=0.3,
    activation='leaky_relu'
)

# Example 3: Minimal CNN (no FC hidden layers)
model3 = CNNModel(
    input_dim=(1, 28, 28),
    num_classes=10,
    conv_layers=[(16, 5, 1, 2), (32, 5, 1, 2)],
    fc_layers=[],
    pool=('max', 2, 2),
    dropout=0.0
)

print("Model 1 (CIFAR-10):", sum(p.numel() for p in model1.parameters()), "parameters")
print("Model 2 (ImageNet-style):", sum(p.numel() for p in model2.parameters()), "parameters")
print("Model 3 (MNIST-style):", sum(p.numel() for p in model3.parameters()), "parameters")