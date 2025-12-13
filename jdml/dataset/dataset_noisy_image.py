import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Literal, Union

NoiseType = Literal["gaussian", "poisson", "speckle", "poisson_gaussian", "poisson-gaussian"]

class NoisyImageDataset(Dataset):
    """
    Wraps clean images (NumPy) and adds noise on the fly.
    Works for grayscale, RGB, or multi-channel data.

    Noise models:
      - "gaussian": y = x + N(0, (sigma/255)^2)
      - "poisson":  y = (1/a) * Poisson(a*x), where a = poisson_scale
      - "speckle":  y = x + x * N(0, (sigma/255)^2)
      - "poisson_gaussian"/"poisson-gaussian":
                     y = (1/a) * Poisson(a*x) + N(0, (sigma/255)^2)
    All computations are done in float32 with image intensities in [0,1].
    """

    def __init__(
            self,
            X: np.ndarray,
            noise_type: NoiseType = "gaussian",
            sigma: float = 25.0,        # for Gaussian parts, given on 0–255 scale (std in pixel units)
            poisson_scale: float = 30,  # 'a' in the model; higher -> more photons -> lower relative shot noise
            seed: Union[int, None] = 0,
            clip: bool = True,
    ):
        self.noise_type = noise_type
        self.sigma = float(sigma)
        self.poisson_scale = float(poisson_scale)
        self.clip = bool(clip)
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Ensure float32 in [0,1]
        X = X.astype(np.float32)
        if X.max() > 1.0:
            X = X / 255.0

        # Ensure shape (N, C, H, W)
        if X.ndim == 3:
            # (N, H, W) → (N, 1, H, W)
            X = X[:, None, :, :]
        elif X.ndim == 4:
            # (N, H, W, C) → (N, C, H, W) for typical images
            if X.shape[-1] <= 4:  # assume channels-last
                X = np.transpose(X, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unsupported shape {X.shape}, expected (N,H,W) or (N,H,W,C)")

        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.X[idx]  # (C, H, W), np.float32 in [0,1]

        # Common conversions
        sigma01 = self.sigma / 255.0
        a = max(self.poisson_scale, 1e-8)  # numerical safety

        if self.noise_type == "gaussian":
            noise = self.rng.normal(0.0, sigma01, size=clean.shape).astype(np.float32)
            noisy = clean + noise

        elif self.noise_type == "poisson":
            lam = np.clip(clean * a, 0.0, None)
            noisy = self.rng.poisson(lam).astype(np.float32) / a

        elif self.noise_type in ("poisson_gaussian", "poisson-gaussian"):
            # Poisson first (shot noise), then additive Gaussian (read noise)
            lam = np.clip(clean * a, 0.0, None)
            shot = self.rng.poisson(lam).astype(np.float32) / a
            read = self.rng.normal(0.0, sigma01, size=clean.shape).astype(np.float32)
            noisy = shot + read

        elif self.noise_type == "speckle":
            mult = self.rng.normal(0.0, sigma01, size=clean.shape).astype(np.float32)
            noisy = clean + clean * mult

        else:
            raise ValueError(f"Unknown noise_type '{self.noise_type}'")

        if self.clip:
            noisy = np.clip(noisy, 0.0, 1.0)

        return torch.from_numpy(noisy), torch.from_numpy(clean)
