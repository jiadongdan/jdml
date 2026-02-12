"""
HDF5-based PyTorch Dataset for patch extraction with augmentation.

Supports lazy loading from HDF5 files for large datasets.
Each image undergoes: Scale → Rotate → Center Crop → Random Patch extraction.

Expected HDF5 structure:
    file.h5
    ├── train/
    │   ├── data   (N, C, H, W) float32, chunked per image
    │   └── labels   (N,) int64 (optional)
    ├── val/
    │   ├── data
    │   └── labels
    └── test/
        ├── data
        └── labels

Usage:
    # Create dataset
    dataset = ScaleRotateCropPatchDataset('data.h5', group_name='train')

    # With DataLoader (num_workers > 0 is safe)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Create HDF5 file with proper chunking
    create_h5_dataset('data.h5', train_images, train_labels, val_images, val_labels)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import h5py
from pathlib import Path


class ScaleRotateCropPatchDataset(Dataset):
    """
    PyTorch Dataset that lazily loads images from HDF5 and extracts
    augmented patches (scale, rotate, crop).

    Each image produces `n_patches` samples per epoch. Images are read
    one at a time from HDF5, so memory usage stays constant regardless
    of dataset size.

    Args:
        h5_file: Path to HDF5 file.
        group_name: Group within HDF5 file ('train', 'val', 'test').
        data_key: Dataset key for images within the group.
        labels_key: Dataset key for labels within the group (set None to disable).
        n_patches: Number of random patches to extract per image per epoch.
        patch_size: Size of square patches.
        random_rotation: If True, randomly rotate images.
        rotation_range: (min_angle, max_angle) in degrees.
        random_scale: If True, randomly scale images.
        scale_range: (min_scale, max_scale) relative to original size.
        interpolation: Interpolation mode ('bilinear', 'bicubic', 'nearest').
        deterministic: If True, use seeded RNG for reproducible augmentation.
        seed: Random seed for deterministic mode.

    Returns per __getitem__:
        patch: torch.FloatTensor of shape (C, patch_size, patch_size)
        label: int label (if labels exist) or image index
    """

    def __init__(
            self,
            h5_file,
            group_name='train',
            data_key='data',
            labels_key='labels',
            n_patches=1,
            patch_size=64,
            random_rotation=True,
            rotation_range=(0, 360),
            random_scale=True,
            scale_range=(0.8, 1.2),
            interpolation='bilinear',
            deterministic=False,
            seed=42,
    ):
        self.h5_path = Path(h5_file)
        self.group_name = group_name
        self.data_key = data_key
        self.labels_key = labels_key

        # --- Read metadata from HDF5 (open briefly, then close) ---
        with h5py.File(self.h5_path, 'r') as f:
            if group_name not in f:
                raise KeyError(
                    f"Group '{group_name}' not found. Available: {list(f.keys())}"
                )
            grp = f[group_name]

            if data_key not in grp:
                raise KeyError(
                    f"Dataset '{data_key}' not found in group '{group_name}'. "
                    f"Available: {list(grp.keys())}"
                )

            shape = grp[data_key].shape
            dtype = grp[data_key].dtype

            # Cache labels in memory (they're small)
            if labels_key is not None and labels_key in grp:
                self.label_cache = grp[labels_key][:]
                self.has_labels = True
            else:
                self.label_cache = None
                self.has_labels = False

        # --- Validate data shape and type ---
        if len(shape) != 4:
            raise ValueError(
                f"Expected 4D array (N, C, H, W), got shape {shape}"
            )
        if dtype != np.float32:
            raise TypeError(
                f"Expected float32 images, got {dtype}"
            )

        self.n_images, self.n_channels, self.image_height, self.image_width = shape

        if self.has_labels and len(self.label_cache) != self.n_images:
            raise ValueError(
                f"Labels length {len(self.label_cache)} != data length {self.n_images}"
            )

        # --- Validate patch size ---
        min_dim = min(self.image_height, self.image_width)
        min_scale = scale_range[0] if random_scale else 1.0
        min_dim_after_transform = int(min_dim * min_scale / np.sqrt(2))
        if min_dim_after_transform < patch_size:
            raise ValueError(
                f"Images too small for patch_size={patch_size}. "
                f"After worst-case transform (scale={min_scale}, rotation crop), "
                f"min usable dimension is ~{min_dim_after_transform}. "
                f"Original size: {self.image_height}x{self.image_width}. "
                f"Options: reduce patch_size, increase scale_range[0], or use larger images."
            )

        # --- Augmentation parameters ---
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.interpolation_name = interpolation

        if rotation_range[0] >= rotation_range[1]:
            raise ValueError(f"Invalid rotation_range: {rotation_range}")
        if not (0 < scale_range[0] < scale_range[1]):
            raise ValueError(f"Invalid scale_range: {scale_range}")

        # Set interpolation mode
        interp_modes = {
            'nearest': TF.InterpolationMode.NEAREST,
            'bilinear': TF.InterpolationMode.BILINEAR,
            'bicubic': TF.InterpolationMode.BICUBIC,
        }
        if interpolation not in interp_modes:
            raise ValueError(
                f"Invalid interpolation '{interpolation}'. Choose from: {list(interp_modes.keys())}"
            )
        self.interpolation = interp_modes[interpolation]

        # --- Deterministic mode ---
        self.deterministic = deterministic
        self.seed = seed
        if deterministic:
            #self.rng = np.random.RandomState(seed)
            # RandomState is legacy. Use Generator instead
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # --- Totals ---
        self.total_samples = self.n_images * n_patches

        # --- HDF5 handle (lazily opened per worker) ---
        self._h5_file = None

    # ------------------------------------------------------------------
    # HDF5 access (worker-safe lazy opening)
    # ------------------------------------------------------------------

    def _get_dataset(self):
        """
        Return the HDF5 dataset handle, opening the file lazily.

        Each DataLoader worker process gets its own file handle,
        which is necessary because HDF5 files are not fork-safe.
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file[self.group_name][self.data_key]

    def close(self):
        """Explicitly close the HDF5 file handle."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Image processing pipeline: Scale → Rotate → Center Crop
    # ------------------------------------------------------------------

    def _process_image(self, image_np):
        """
        Process a single image: Scale → Rotate → Center Crop.

        Args:
            image_np: numpy array (C, H, W) float32

        Returns:
            Cropped torch.FloatTensor (C, H', W')
        """
        img = torch.from_numpy(image_np.copy())

        # Step 1: Random scaling
        if self.random_scale:
            scale = float(self.rng.uniform(self.scale_range[0], self.scale_range[1]))
            _, H, W = img.shape
            new_h = max(1, int(H * scale))
            new_w = max(1, int(W * scale))
            img = TF.resize(
                img, [new_h, new_w],
                interpolation=self.interpolation,
                antialias=True,
            )

        # Step 2: Random rotation
        if self.random_rotation:
            angle = float(self.rng.uniform(self.rotation_range[0], self.rotation_range[1]))
            img = TF.rotate(
                img, angle,
                interpolation=self.interpolation,
                expand=False,
            )

        # Step 3: Center crop to remove black corners from rotation
        _, H, W = img.shape
        crop_h = int(H / np.sqrt(2))
        crop_w = int(W / np.sqrt(2))
        img = TF.center_crop(img, [crop_h, crop_w])

        return img

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _extract_random_patch(self, image, patch_size):
        """
        Extract a random square patch from the processed image.

        If the image is smaller than patch_size (edge case after aggressive
        downscaling), it is resized up with a small margin first.

        Args:
            image: torch.FloatTensor (C, H, W)
            patch_size: int

        Returns:
            torch.FloatTensor (C, patch_size, patch_size)
        """
        _, H, W = image.shape

        # Safety resize if image is too small
        if H < patch_size or W < patch_size:
            scale = max(patch_size / H, patch_size / W) * 1.1
            image = TF.resize(
                image,
                [max(patch_size, int(H * scale)),
                 max(patch_size, int(W * scale))],
                interpolation=self.interpolation,
                antialias=True,
            )
            _, H, W = image.shape

        top = int(self.rng.integers(0, H - patch_size + 1))
        left = int(self.rng.integers(0, W - patch_size + 1))

        return image[:, top:top + patch_size, left:left + patch_size]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Get a single augmented patch.

        Returns:
            patch: torch.FloatTensor (C, patch_size, patch_size)
            label: int (class label if available, else image index)
        """
        image_idx = idx // self.n_patches

        # Lazy read from HDF5 — only one image loaded at a time
        image_np = self._get_dataset()[image_idx]  # (C, H, W) float32

        # Augment and crop
        processed = self._process_image(image_np)

        # Extract patch
        patch = self._extract_random_patch(processed, self.patch_size)

        # Label
        if self.has_labels:
            label = int(self.label_cache[image_idx])
            return patch, label
        else:
            return patch, image_idx

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self):
        lines = [
            f"{self.__class__.__name__}(",
            f"  h5_file='{self.h5_path}',",
            f"  group='{self.group_name}',",
            f"  num_images={self.n_images},",
            f"  image_shape=({self.n_channels}, {self.image_height}, {self.image_width}),",
            f"  total_patches={self.total_samples},",
            f"  n_patches_per_image={self.n_patches},",
            f"  patch_size={self.patch_size}x{self.patch_size},",
            f"  has_labels={self.has_labels},",
        ]

        aug_info = []
        if self.random_rotation:
            aug_info.append(f"rotation{self.rotation_range}")
        if self.random_scale:
            aug_info.append(f"scale{self.scale_range}")
        lines.append(f"  augmentations=[{', '.join(aug_info)}]," if aug_info else "  augmentations=None,")

        lines.append(f"  interpolation='{self.interpolation_name}',")
        lines.append(f"  deterministic={self.deterministic}" + (f" (seed={self.seed})" if self.deterministic else ""))
        lines.append(")")
        return "\n".join(lines)
