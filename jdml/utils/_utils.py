import torch.nn as nn
import torch.optim as optim

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