import math
import h5py
import numpy as np
# torch related
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def load_h5_data(data_path):
    with h5py.File(data_path, 'r') as f:
        X_train = f['train']['X'][:].astype(np.float32)
        y_train = f['train']['y'][:].astype(np.int64)
        X_test  = f['test']['X'][:].astype(np.float32)
        y_test  = f['test']['y'][:].astype(np.int64)
    print(f"Loaded X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Loaded X_test  {X_test.shape},  y_test  {y_test.shape}")
    return X_train, y_train, X_test, y_test

def _rotate_any_angle(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    x: (C,H,W) tensor on CPU
    angle_deg: rotation in degrees, positive is counterclockwise
    Returns a rotated (C,H,W) tensor, same spatial size.
    """
    C, H, W = x.shape
    # Build 2x3 affine matrix for rotation about the image center
    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    theta = x.new_tensor([[cos_a, -sin_a, 0.0],
                          [sin_a,  cos_a, 0.0]])  # (2,3)
    theta = theta.unsqueeze(0)                     # (1,2,3)

    # Grid for a single sample; normalize coords in [-1,1]
    grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)
    x_b = x.unsqueeze(0)                           # (1,C,H,W)
    y = F.grid_sample(x_b, grid,
                      mode='bilinear',             # smooth for arbitrary angles
                      padding_mode='zeros',
                      align_corners=False)
    return y.squeeze(0)

class AugmentedTensorDataset(Dataset):
    """
    Wraps (X, y) tensors and applies random rotation at fetch time.
    - rotate_mode: 'off' | 'k90' (0/90/180/270) | 'any' (uniform in [-degrees,+degrees])
    - p: probability of applying a rotation
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 rotate_mode: str = 'any', p: float = 1.0, degrees: float = 180.0):
        assert rotate_mode in ('off', 'k90', 'any')
        self.X, self.y = X, y
        self.rotate_mode = rotate_mode
        self.p = float(p)
        self.degrees = float(degrees)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (C,H,W) float32
        label = self.y[idx]

        if self.rotate_mode != 'off' and torch.rand(()) < self.p:
            if self.rotate_mode == 'k90':
                # exact 90° multiples, no interpolation artifacts
                k = int(torch.randint(0, 4, (1,)))
                x = torch.rot90(x, k=k, dims=(1, 2))
            else:
                # arbitrary small rotation within [-degrees, +degrees]
                angle = (torch.rand(()) * 2.0 - 1.0).item() * self.degrees
                x = _rotate_any_angle(x, angle)

        return x, label


class SingleChannelDataset(Dataset):
    def __init__(self, X, y, rgb=False, device=None):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (N, C, H, W).
            y (torch.Tensor): Labels of shape (N,).
            rgb (bool): If True, repeat first channel 3 times to simulate RGB.
            device (torch.device or str, optional): Move tensors to this device.
        """
        assert X.ndim == 4, f"Expected X of shape (N, C, H, W), got {X.shape}"
        self.X = X[:, 0:1, :, :]  # keep only the first channel
        if rgb:
            self.X = self.X.repeat(1, 3, 1, 1)  # repeat to 3 channels

        self.y = y
        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
