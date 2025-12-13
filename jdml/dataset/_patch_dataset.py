import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path


class ScaleRotateCropPatchDataset(Dataset):
    def __init__(self, data, labels=None, n_patches=1, patch_size=64,
                 random_rotation=True, rotation_range=(0, 360),
                 random_scale=True, scale_range=(0.8, 1.2),
                 interpolation='bilinear',
                 deterministic=False, seed=42):
        """
        PyTorch Dataset for rotation, scaling, cropping, and patch extraction.

        Args:
            data: Either:
                  - numpy array of shape (N, C, H, W) in float32
                  - path to .npy file (will be memory-mapped)
            labels: numpy array of shape (N,) for classification, or None
            n_patches: number of random patches to extract per image per epoch
            patch_size: size of square patches (default: 64)
            random_rotation: if True, randomly rotate (default: True)
            rotation_range: tuple (min_angle, max_angle) in degrees (default: (0, 360))
            random_scale: if True, randomly scale the image (default: True)
            scale_range: tuple (min_scale, max_scale), e.g., (0.8, 1.2) for ±20%
            interpolation: interpolation mode - 'bilinear', 'bicubic', or 'nearest'
            deterministic: if True, use deterministic random patch extraction (for validation/testing)
            seed: random seed for deterministic mode

        Returns:
            patch: torch.FloatTensor of shape (C, patch_size, patch_size)
            label: label (if labels provided) or image_idx (if no labels)
        """
        # Handle memory mapping
        if isinstance(data, (str, Path)):
            self.data = np.load(data, mmap_mode='r')
            self.is_mmap = True
            self.data_path = str(data)
        else:
            self.data = data
            self.is_mmap = False
            self.data_path = None

        # Error handling - validate data
        assert len(self.data.shape) == 4, \
            f"Expected 4D array (N, C, H, W), got shape {self.data.shape}"
        assert self.data.dtype == np.float32, \
            f"Expected float32, got {self.data.dtype}"

        N, C, H, W = self.data.shape
        self.n_channels = C
        self.image_height = H
        self.image_width = W

        # Validate patch size
        min_dim_after_crop = int(min(H, W) / np.sqrt(2))
        assert min_dim_after_crop >= patch_size, \
            f"Images too small for patch_size={patch_size}. " \
            f"After rotation crop, min dimension is ~{min_dim_after_crop}. " \
            f"Original size: {H}x{W}"

        # Handle labels
        self.labels = labels
        if labels is not None:
            assert len(labels) == N, \
                f"Labels length {len(labels)} doesn't match data length {N}"
            self.has_labels = True
        else:
            self.has_labels = False

        # Store parameters
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_scale = random_scale
        self.scale_range = scale_range
        self.interpolation_name = interpolation

        # Validate ranges
        assert rotation_range[0] < rotation_range[1], \
            f"Invalid rotation_range: {rotation_range}"
        assert scale_range[0] < scale_range[1] and scale_range[0] > 0, \
            f"Invalid scale_range: {scale_range}"

        # Set interpolation mode
        interp_modes = {
            'nearest': TF.InterpolationMode.NEAREST,
            'bilinear': TF.InterpolationMode.BILINEAR,
            'bicubic': TF.InterpolationMode.BICUBIC,
        }
        assert interpolation in interp_modes, \
            f"Invalid interpolation '{interpolation}'. Choose from: {list(interp_modes.keys())}"
        self.interpolation = interp_modes[interpolation]

        # Deterministic mode
        self.deterministic = deterministic
        self.seed = seed
        if deterministic:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

        self.n_images = len(self.data)
        self.total_samples = self.n_images * n_patches

    def __len__(self):
        return self.total_samples

    def __repr__(self):
        """String representation of the dataset."""
        lines = [
            f"{self.__class__.__name__}(",
            f"  num_images={self.n_images},",
            f"  image_shape=({self.n_channels}, {self.image_height}, {self.image_width}),",
            f"  total_patches={self.total_samples},",
            f"  n_patches_per_image={self.n_patches},",
            f"  patch_size={self.patch_size}x{self.patch_size},",
            f"  has_labels={self.has_labels},",
            f"  memory_mapped={self.is_mmap}",
        ]

        if self.is_mmap:
            lines.append(f"    (path: {self.data_path}),")

        # Augmentation info
        aug_info = []
        if self.random_rotation:
            aug_info.append(f"rotation{self.rotation_range}")
        if self.random_scale:
            aug_info.append(f"scale{self.scale_range}")

        if aug_info:
            lines.append(f"  augmentations=[{', '.join(aug_info)}],")
        else:
            lines.append(f"  augmentations=None,")

        lines.append(f"  interpolation='{self.interpolation_name}',")
        lines.append(f"  deterministic={self.deterministic}")

        if self.deterministic:
            lines.append(f"    (seed={self.seed})")

        lines.append(")")

        return "\n".join(lines)

    def _process_image(self, image):
        """
        Process image: Scale → Rotate → Crop

        Args:
            image: numpy array of shape (C, H, W) in float32

        Returns:
            cropped image as torch.FloatTensor (C, H', W')
        """
        # Convert to torch tensor (preserves float32)
        img_tensor = torch.from_numpy(image.copy() if self.is_mmap else image)

        # Step 1: Random scaling
        if self.random_scale:
            scale_factor = self.rng.uniform(self.scale_range[0], self.scale_range[1])
            _, H, W = img_tensor.shape
            new_h = int(H * scale_factor)
            new_w = int(W * scale_factor)
            scaled = TF.resize(
                img_tensor, [new_h, new_w],
                interpolation=self.interpolation,
                antialias=True
            )
        else:
            scaled = img_tensor

        # Step 2: Random rotation
        if self.random_rotation:
            angle = self.rng.uniform(self.rotation_range[0], self.rotation_range[1])
            rotated = TF.rotate(
                scaled,
                float(angle),
                interpolation=self.interpolation,
                expand=False
            )
        else:
            rotated = scaled

        # Step 3: Center crop (removes black corners from rotation)
        _, H, W = rotated.shape
        new_h = int(H / np.sqrt(2))
        new_w = int(W / np.sqrt(2))
        cropped = TF.center_crop(rotated, [new_h, new_w])

        return cropped

    def _extract_random_patch(self, image, patch_size):
        """
        Extract a random patch from the image.

        Args:
            image: torch.FloatTensor of shape (C, H, W)
            patch_size: size of the square patch

        Returns:
            patch of shape (C, patch_size, patch_size)
        """
        _, H, W = image.shape

        # Ensure image is large enough for patch extraction
        if H < patch_size or W < patch_size:
            scale = max(patch_size / H, patch_size / W) * 1.1
            new_h = int(H * scale)
            new_w = int(W * scale)
            image = TF.resize(
                image, [new_h, new_w],
                interpolation=self.interpolation,
                antialias=True
            )
            _, H, W = image.shape

        # Random crop location (using deterministic RNG if specified)
        top = self.rng.randint(0, H - patch_size + 1)
        left = self.rng.randint(0, W - patch_size + 1)

        patch = image[:, top:top+patch_size, left:left+patch_size]
        return patch

    def __getitem__(self, idx):
        """
        Get a single patch.

        Returns:
            patch: torch.FloatTensor (C, patch_size, patch_size)
            label: classification label (if labels provided) or image_idx
        """
        # Determine which image this patch belongs to
        image_idx = idx // self.n_patches

        # Get the original image (already in C, H, W format, float32)
        original_image = self.data[image_idx]

        # Process the image (scale, rotate, crop)
        processed_image = self._process_image(original_image)

        # Extract random patch
        patch = self._extract_random_patch(processed_image, self.patch_size)

        # Return patch with label or image index
        if self.labels is not None:
            label = self.labels[image_idx]
            return patch, label
        else:
            return patch, image_idx

