import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    """
    Flexible CNN model constructor with advanced features.

    Args:
        input_dim: tuple (channels, height, width) - input image dimensions
        num_classes: int - number of output classes
        conv_layers: list of tuples defining convolutional layers
            Each tuple: (out_channels, kernel_size, stride, padding, pool)
            pool: True/False - whether to apply pooling after this layer
            Example: [(32, 3, 1, 1, True), (64, 3, 1, 1, False), (64, 3, 1, 1, True)]
        fc_layers: list of ints defining fully connected layer sizes
            Example: [512, 256] creates two FC layers
        pool_config: tuple (type, kernel_size, stride) - pooling configuration
            type: 'max' or 'avg'
            Example: ('max', 2, 2)
        use_gap: bool - use Global Average Pooling instead of flatten + FC
        dropout: float - dropout rate (0 to disable)
        activation: str - 'relu', 'leaky_relu', 'elu', 'selu'

    Example usage:
        model = CNNModel(
            input_dim=(3, 32, 32),
            num_classes=10,
            conv_layers=[(32, 3, 1, 1, True), (64, 3, 1, 1, False), (128, 3, 1, 1, True)],
            fc_layers=[512, 256],
            pool_config=('max', 2, 2),
            use_gap=False,
            dropout=0.5
        )

        # View model architecture
        model.summary()

        # Get optimizer and loss
        optimizer = model.get_optimizer('adam', lr=0.001)
        criterion = model.get_loss('crossentropy')
    """
    def __init__(self, input_dim, num_classes, conv_layers, fc_layers=None,
                 pool_config=('max', 2, 2), use_gap=False, dropout=0.5, activation='relu'):
        super(CNNModel, self).__init__()

        self.input_channels, self.input_h, self.input_w = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.use_gap = use_gap

        # Validate input dimensions
        if self.input_h <= 0 or self.input_w <= 0 or self.input_channels <= 0:
            raise ValueError(f"Invalid input dimensions: {input_dim}")

        # Build activation function
        self.activation = self._get_activation(activation)

        # Build pooling layer
        if pool_config[0] == 'max':
            self.pool = nn.MaxPool2d(pool_config[1], pool_config[2])
        elif pool_config[0] == 'avg':
            self.pool = nn.AvgPool2d(pool_config[1], pool_config[2])
        else:
            raise ValueError(f"Unknown pooling type: {pool_config[0]}")

        # Build convolutional layers
        self.conv_blocks = nn.ModuleList()
        self.pool_flags = []
        in_channels = self.input_channels

        for layer_config in conv_layers:
            out_channels, kernel_size, stride, padding, use_pool = layer_config

            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                self._get_activation(activation)
            )
            self.conv_blocks.append(block)
            self.pool_flags.append(use_pool)
            in_channels = out_channels

        # Validate feature map dimensions
        final_h, final_w = self._calculate_feature_dims(conv_layers, pool_config)
        if final_h <= 0 or final_w <= 0:
            raise ValueError(
                f"Feature map dimensions became invalid ({final_h}x{final_w}). "
                f"Try reducing pooling or using smaller strides."
            )

        # Global Average Pooling or Flatten
        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.flat_features = conv_layers[-1][0]  # Just the channels
            if fc_layers:
                print("Warning: fc_layers ignored when use_gap=True. GAP directly connects to classifier.")
            fc_layers = []  # GAP goes directly to classifier
        else:
            self.gap = None
            self.flat_features = self._calculate_flat_features(conv_layers, pool_config)

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

        # Store config for summary
        self.conv_layers = conv_layers
        self.fc_layers_config = fc_layers
        self.pool_config = pool_config

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        return activations.get(name, nn.ReLU())

    def _calculate_feature_dims(self, conv_layers, pool_config):
        """Calculate final feature map dimensions"""
        h, w = self.input_h, self.input_w

        for out_channels, kernel_size, stride, padding, use_pool in conv_layers:
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1

            if use_pool:
                h = h // pool_config[2]
                w = w // pool_config[2]

        return h, w

    def _calculate_flat_features(self, conv_layers, pool_config):
        """Calculate the flattened feature size after conv and pool layers"""
        h, w = self._calculate_feature_dims(conv_layers, pool_config)
        return conv_layers[-1][0] * h * w

    def forward(self, x):
        # Convolutional blocks with conditional pooling
        for conv_block, use_pool in zip(self.conv_blocks, self.pool_flags):
            x = conv_block(x)
            if use_pool:
                x = self.pool(x)

        # Global Average Pooling or Flatten
        if self.use_gap:
            x = self.gap(x)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)

        # Fully connected blocks
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # Classification layer
        x = self.classifier(x)
        return x

    def summary(self, input_size=None):
        """Print model architecture summary"""
        if input_size is None:
            input_size = (self.input_channels, self.input_h, self.input_w)

        print("=" * 80)
        print(f"CNN Model Architecture")
        print("=" * 80)
        print(f"Input: {input_size}")
        print(f"Output Classes: {self.num_classes}")
        print("-" * 80)

        # Conv layers
        print("Convolutional Layers:")
        h, w = self.input_h, self.input_w
        in_ch = self.input_channels

        for i, (out_ch, k, s, p, pool) in enumerate(self.conv_layers):
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1
            print(f"  Conv{i+1}: {in_ch} -> {out_ch}, kernel={k}, stride={s}, padding={p}")
            print(f"          Output: ({out_ch}, {h}, {w})")

            if pool:
                h = h // self.pool_config[2]
                w = w // self.pool_config[2]
                print(f"  Pool{i+1}: {self.pool_config[0]}, kernel={self.pool_config[1]}, stride={self.pool_config[2]}")
                print(f"          Output: ({out_ch}, {h}, {w})")
            in_ch = out_ch

        print("-" * 80)

        # GAP or Flatten
        if self.use_gap:
            print(f"Global Average Pooling: ({in_ch}, {h}, {w}) -> ({in_ch}, 1, 1)")
            print(f"Flatten: {in_ch}")
        else:
            print(f"Flatten: ({in_ch}, {h}, {w}) -> {in_ch * h * w}")

        print("-" * 80)

        # FC layers
        if self.fc_layers_config:
            print("Fully Connected Layers:")
            in_features = self.flat_features
            for i, hidden in enumerate(self.fc_layers_config):
                print(f"  FC{i+1}: {in_features} -> {hidden} (dropout={self.dropout_rate})")
                in_features = hidden
            print(f"  Classifier: {in_features} -> {self.num_classes}")
        else:
            print(f"Classifier: {self.flat_features} -> {self.num_classes}")

        print("-" * 80)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 80)



# Example 1: Standard CNN
print("\n=== Example 1: Standard CNN ===")
model1 = CNNModel(
    input_dim=(3, 32, 32),
    num_classes=10,
    conv_layers=[(32, 3, 1, 1, True), (64, 3, 1, 1, False), (128, 3, 1, 1, True)],
    fc_layers=[512, 256],
    pool_config=('max', 2, 2),
    dropout=0.5
)
model1.summary()

# Example 2: With Global Average Pooling (fewer parameters)
print("\n=== Example 2: With Global Average Pooling ===")
model2 = CNNModel(
    input_dim=(3, 32, 32),
    num_classes=10,
    conv_layers=[(32, 3, 1, 1, True), (64, 3, 1, 1, True), (128, 3, 1, 1, True)],
    fc_layers=None,  # Not used with GAP
    pool_config=('max', 2, 2),
    use_gap=True,
    dropout=0.5
)
model2.summary()

# Example usage with optimizer and loss
optimizer = model1.get_optimizer('adam', lr=0.001, weight_decay=1e-4)
criterion = model1.get_loss('crossentropy')
print(f"\nOptimizer: {optimizer.__class__.__name__}")
print(f"Loss: {criterion.__class__.__name__}")