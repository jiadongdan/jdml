import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


# Utility Functions
def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0, **kwargs):
    """
    Get optimizer for any PyTorch model

    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'sgd', 'rmsprop', 'adamw'
        lr: learning rate
        weight_decay: L2 regularization
        **kwargs: additional optimizer arguments (e.g., momentum for SGD)

    Returns:
        PyTorch optimizer

    Example:
        optimizer = get_optimizer(model, 'adam', lr=0.001, weight_decay=1e-4)
        optimizer = get_optimizer(model, 'sgd', lr=0.01, momentum=0.9)
    """
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adamw': optim.AdamW
    }

    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

    optimizer_class = optimizers[optimizer_name.lower()]

    # SGD-specific defaults
    if optimizer_name.lower() == 'sgd':
        kwargs.setdefault('momentum', 0.9)

    return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def get_loss(loss_name='crossentropy', **kwargs):
    """
    Get loss function

    Args:
        loss_name: 'crossentropy', 'bce', 'mse', 'l1'
        **kwargs: additional loss arguments (e.g., weight, reduction)

    Returns:
        PyTorch loss function

    Example:
        criterion = get_loss('crossentropy')
        criterion = get_loss('crossentropy', weight=torch.tensor([1.0, 2.0, 1.5]))
    """
    losses = {
        'crossentropy': nn.CrossEntropyLoss,
        'bce': nn.BCEWithLogitsLoss,
        'mse': nn.MSELoss,
        'l1': nn.L1Loss
    }

    if loss_name.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(losses.keys())}")

    return losses[loss_name.lower()](**kwargs)


def train_model(model, train_loader, val_loader, num_epochs, optimizer, criterion,
                device='cuda', checkpoint_dir='checkpoints', use_tqdm=True, resume_from=None,
                scheduler=None):
    """
    Training loop with progress bars and model checkpointing

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: total number of epochs to train (not additional epochs)
        optimizer: PyTorch optimizer
        criterion: loss function
        device: 'cuda' or 'cpu'
        checkpoint_dir: directory to save model checkpoints
        use_tqdm: if True, show progress bars; if False, print simple progress
        resume_from: path to checkpoint to resume training from (optional)
        scheduler: learning rate scheduler (optional)
            Supports: ReduceLROnPlateau, StepLR, CosineAnnealingLR, etc.

    Saves two checkpoints:
        - 'best_model.pth': saved when validation loss improves (best for inference)
        - 'last_checkpoint.pth': saved every epoch (for resuming training)

    Returns:
        dict with training history (train_loss, val_loss, train_acc, val_acc, lr)

    Example:
        # Train from scratch with scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        optimizer = get_optimizer(model, 'adam', lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device='cuda'
        )

        # Resume training (scheduler state is also restored)
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device='cuda',
            resume_from='checkpoints/last_checkpoint.pth'
        )
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        print(f"Resuming training from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])

        # Resume scheduler state if it exists
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored")

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best val loss: {best_val_loss:.4f}")
        print(f"Will train until epoch {num_epochs}")
        print("-" * 60)

        if start_epoch >= num_epochs:
            print(f"Warning: start_epoch ({start_epoch}) >= num_epochs ({num_epochs})")
            print("No training will be performed. Increase num_epochs to continue training.")
            return {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'lr': []
            }

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm or simple iteration
        train_iter = tqdm(train_loader, desc='Training', leave=False) if use_tqdm else train_loader

        for batch_idx, (inputs, labels) in enumerate(train_iter):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress
            if use_tqdm:
                train_iter.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })
            elif batch_idx % 10 == 0:  # Print every 10 batches if no tqdm
                print(f'  Batch [{batch_idx}/{len(train_loader)}] - '
                      f'Loss: {loss.item():.4f}, Acc: {100 * train_correct / train_total:.2f}%')

        # Calculate average training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Use tqdm or simple iteration
        val_iter = tqdm(val_loader, desc='Validation', leave=False) if use_tqdm else val_loader

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_iter):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress
                if use_tqdm:
                    val_iter.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * val_correct / val_total:.2f}%'
                    })

        # Calculate average validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100 * val_correct / val_total

        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Print epoch summary
        print(f'Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

        # Learning rate scheduler step
        if scheduler is not None:
            # ReduceLROnPlateau needs validation loss as input
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        # Save last checkpoint (every epoch)
        last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_acc': epoch_train_acc,
            'val_acc': epoch_val_acc,
            'best_val_loss': best_val_loss
        }

        # Save scheduler state if it exists
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint_data, last_checkpoint_path)

        # Save best model (when validation loss improves)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            best_checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc
            }

            # Save scheduler state if it exists
            if scheduler is not None:
                best_checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(best_checkpoint_data, best_checkpoint_path)
            print(f'✓ New best model saved! (Val Loss: {epoch_val_loss:.4f})')


    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print('=' * 60)

    return history


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load model from checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: path to checkpoint file
        optimizer: (optional) optimizer to load state
        device: 'cuda' or 'cpu'

    Returns:
        dict with checkpoint info (epoch, losses, accuracies)

    Example:
        info = load_checkpoint(model, 'checkpoints/best_model.pth', optimizer)
        print(f"Loaded model from epoch {info['epoch']}")
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f} | Train Acc: {checkpoint['train_acc']:.2f}%")
    print(f"Val Loss: {checkpoint['val_loss']:.4f} | Val Acc: {checkpoint['val_acc']:.2f}%")

    return {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'train_acc': checkpoint['train_acc'],
        'val_acc': checkpoint['val_acc']
    }

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



    @classmethod
    def from_config(cls, config):
        """Create model from configuration dictionary"""
        return cls(**config)

    def get_config(self):
        """Get model configuration as dictionary"""
        return {
            'input_dim': (self.input_channels, self.input_h, self.input_w),
            'num_classes': self.num_classes,
            'conv_layers': self.conv_layers,
            'fc_layers': self.fc_layers_config,
            'pool_config': self.pool_config,
            'use_gap': self.use_gap,
            'dropout': self.dropout_rate
        }


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

# Example usage with training
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

# Prepare your data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create model and training components
model = CNNModel(
    input_dim=(3, 32, 32),
    num_classes=10,
    conv_layers=[(32, 3, 1, 1, True), (64, 3, 1, 1, True), (128, 3, 1, 1, True)],
    fc_layers=[512, 256],
    pool_config=('max', 2, 2),
    dropout=0.5
)

optimizer = get_optimizer(model, 'adam', lr=0.001, weight_decay=1e-4)
criterion = get_loss('crossentropy')

# Option 1: ReduceLROnPlateau (reduces LR when val loss plateaus)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Option 2: StepLR (reduces LR every N epochs)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Option 3: CosineAnnealingLR (cosine annealing)
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Train from scratch with scheduler
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device='cuda'
)

# Resume training - scheduler state is also restored
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,  # Total epochs
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device='cuda',
    resume_from='checkpoints/last_checkpoint.pth'
)

# Plot learning rate schedule
import matplotlib.pyplot as plt
plt.plot(history['lr'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()

# Load best model for inference
load_checkpoint(model, 'checkpoints/best_model.pth')
"""

optimizer = get_optimizer(model1, 'adam', lr=0.001, weight_decay=1e-4)
criterion = get_loss('crossentropy')
print(f"\nOptimizer: {optimizer.__class__.__name__}")
print(f"Loss: {criterion.__class__.__name__}")

# Can use with any PyTorch model
optimizer_sgd = get_optimizer(model2, 'sgd', lr=0.01, momentum=0.9, nesterov=True)
print(f"SGD Optimizer: {optimizer_sgd.__class__.__name__}")