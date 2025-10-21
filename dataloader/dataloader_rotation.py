import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------------------------
# Rotation utility (same as before)
# ----------------------------------------
def _rotate_any_angle(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """
    x: (C,H,W) tensor on CPU
    angle_deg: rotation in degrees, positive is counterclockwise
    Returns a rotated (C,H,W) tensor, same spatial size.
    """
    C, H, W = x.shape
    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    theta = x.new_tensor([[cos_a, -sin_a, 0.0],
                          [sin_a,  cos_a, 0.0]]).unsqueeze(0)  # (1,2,3)
    grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)
    y = F.grid_sample(x.unsqueeze(0), grid,
                      mode='bilinear',
                      padding_mode='zeros',
                      align_corners=False)
    return y.squeeze(0)


# ----------------------------------------
# Disk mask helper
# ----------------------------------------
def _make_disk_mask(H: int, W: int, device=None) -> torch.Tensor:
    """
    Return a (1,H,W) float mask with 1 inside a centered disk, 0 outside.
    """
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device),
                            indexing='ij')
    cy, cx = H / 2, W / 2
    r = min(H, W) / 2
    mask = ((yy - cy)**2 + (xx - cx)**2 <= r**2).float()
    return mask.unsqueeze(0)  # (1,H,W)


# ----------------------------------------
# Dataset with rotation + optional mask
# ----------------------------------------
class AugmentedTensorDataset(Dataset):
    """
    Wraps (X, y) tensors and applies random rotation and/or disk mask.
    - rotate_mode: 'off' | 'k90' | 'any'
    - p: probability of rotation
    - apply_mask: if True, multiply by a centered disk mask
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor,
                 rotate_mode: str = 'any', p: float = 1.0, degrees: float = 180.0,
                 apply_mask: bool = True):
        assert rotate_mode in ('off', 'k90', 'any')
        self.X, self.y = X, y
        self.rotate_mode = rotate_mode
        self.p = float(p)
        self.degrees = float(degrees)
        self.apply_mask = apply_mask
        self._mask_cache = None  # lazily initialized

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (C,H,W)
        label = self.y[idx]

        # --- Random rotation ---
        if self.rotate_mode != 'off' and torch.rand(()) < self.p:
            if self.rotate_mode == 'k90':
                k = int(torch.randint(0, 4, (1,)))
                x = torch.rot90(x, k=k, dims=(1, 2))
            else:
                angle = (torch.rand(()) * 2.0 - 1.0).item() * self.degrees
                x = _rotate_any_angle(x, angle)

        # --- Apply disk mask ---
        if self.apply_mask:
            if self._mask_cache is None or self._mask_cache.shape[1:] != x.shape[1:]:
                self._mask_cache = _make_disk_mask(x.shape[1], x.shape[2], device=x.device)
            x = x * self._mask_cache

        return x, label