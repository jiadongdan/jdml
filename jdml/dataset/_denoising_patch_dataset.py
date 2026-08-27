"""Self-contained rotate/crop/rescale dataset for Atomic S/TEM denoising.

This module consolidates the original ``_denoising_patch_dataset.py`` and its
three local dependencies (``_geometry.py``, ``_standard_h5.py``, and the
required parts of ``noise.py``).  Only third-party packages are imported.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset


STANDARD_SCHEMA_VERSION = "atomic-stem-denoise-h5-v1"
LEGACY_CLEAN_MASTER_SCHEMA_VERSION = "clean-master-h5-v1"
DEFAULT_DEQUANTIZE_SCALE = 1.0 / 32767.0
DEFAULT_DEQUANTIZE_OFFSET = 0.0

GAUSSIAN_SIGMA_RANGE: tuple[float, float] = (0.01, 0.25)
POISSON_PEAK_SIGMA_RANGE: tuple[float, float] = (0.12, 0.35)
SCAN_JITTER_SIGMA_RANGE: tuple[float, float] = (0.0, 0.8)


class StandardH5Source:
    """Worker-safe reader for the combined Atomic S/TEM denoising HDF5 file."""

    def __init__(
        self,
        h5_file: str | Path,
        *,
        split: str = "train",
        rdcc_nbytes: int | None = None,
        rdcc_nslots: int | None = None,
        rdcc_w0: float | None = None,
    ) -> None:
        self.h5_path = Path(h5_file)
        self.split = str(split)
        self.rdcc_nbytes = None if rdcc_nbytes is None else int(rdcc_nbytes)
        self.rdcc_nslots = None if rdcc_nslots is None else int(rdcc_nslots)
        self.rdcc_w0 = None if rdcc_w0 is None else float(rdcc_w0)
        if self.rdcc_nbytes is not None and self.rdcc_nbytes < 1:
            raise ValueError("rdcc_nbytes must be >= 1")
        if self.rdcc_nslots is not None and self.rdcc_nslots < 1:
            raise ValueError("rdcc_nslots must be >= 1")
        if self.rdcc_w0 is not None and not 0.0 <= self.rdcc_w0 <= 1.0:
            raise ValueError("rdcc_w0 must be in [0,1]")
        self._h5_file: h5py.File | None = None
        self._h5_pid: int | None = None

        with h5py.File(self.h5_path, "r") as handle:
            self._validate_handle(handle, self.split)
            group = handle[self.split]
            images = group["images"]
            self.n_images = int(images.shape[0])
            self.image_height = int(images.shape[2])
            self.image_width = int(images.shape[3])
            self.mask_names = tuple(sorted(group.get("masks", {}).keys()))
            self.schema_version = str(handle.attrs["schema_version"])
            self.dequantize_scale = float(
                handle.attrs.get("dequantize_scale", DEFAULT_DEQUANTIZE_SCALE)
            )
            self.dequantize_offset = float(
                handle.attrs.get("dequantize_offset", DEFAULT_DEQUANTIZE_OFFSET)
            )
            self.dequantization_source = (
                "root_attrs"
                if "dequantize_scale" in handle.attrs
                else "legacy_fixed_int16_contract"
            )

    @staticmethod
    def _validate_handle(handle: h5py.File, split: str) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of 'train', 'valid', or 'test'")
        for required_split in ("train", "valid", "test"):
            if required_split not in handle:
                raise KeyError(
                    f"standard HDF5 is missing required group '/{required_split}'"
                )
        group = handle[split]
        if "images" not in group:
            raise KeyError(
                f"standard HDF5 is missing required dataset '/{split}/images'"
            )
        images = group["images"]
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(
                f"standard HDF5 '/{split}/images' must have shape [N,1,H,W], "
                f"got {images.shape}"
            )
        if images.dtype != np.int16:
            raise TypeError(
                f"standard HDF5 '/{split}/images' must use int16 storage, "
                f"got {images.dtype}"
            )
        if "schema_version" not in handle.attrs:
            raise KeyError("standard HDF5 is missing required root attr 'schema_version'")
        schema_version = str(handle.attrs["schema_version"])
        if schema_version not in {
            STANDARD_SCHEMA_VERSION,
            LEGACY_CLEAN_MASTER_SCHEMA_VERSION,
        }:
            raise ValueError(
                "unsupported standard HDF5 schema_version: "
                f"{handle.attrs['schema_version']!r}"
            )
        has_scale = "dequantize_scale" in handle.attrs
        has_offset = "dequantize_offset" in handle.attrs
        if has_scale != has_offset:
            raise KeyError(
                "standard HDF5 must provide both dequantize_scale and "
                "dequantize_offset"
            )
        if schema_version == STANDARD_SCHEMA_VERSION and not has_scale:
            raise KeyError(
                "canonical standard HDF5 requires dequantize_scale and "
                "dequantize_offset root attrs"
            )
        scale = float(handle.attrs.get("dequantize_scale", DEFAULT_DEQUANTIZE_SCALE))
        offset = float(handle.attrs.get("dequantize_offset", DEFAULT_DEQUANTIZE_OFFSET))
        if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(offset):
            raise ValueError("invalid HDF5 dequantization scale/offset")
        if "metadata/json" not in group:
            raise KeyError(f"standard HDF5 is missing required '/{split}/metadata/json'")
        if group["metadata/json"].shape != (images.shape[0],):
            raise ValueError(
                f"'/{split}/metadata/json' length must match '/{split}/images'"
            )
        if "masks" in group:
            for name, mask in group["masks"].items():
                if mask.shape != images.shape:
                    raise ValueError(
                        f"mask {name!r} has shape {mask.shape}, expected {images.shape}"
                    )
                if mask.dtype != np.uint8:
                    raise TypeError(f"mask {name!r} must use uint8, got {mask.dtype}")

    def __len__(self) -> int:
        return self.n_images

    def _open_kwargs(self) -> dict[str, int | float]:
        kwargs: dict[str, int | float] = {}
        if self.rdcc_nbytes is not None:
            kwargs["rdcc_nbytes"] = self.rdcc_nbytes
        if self.rdcc_nslots is not None:
            kwargs["rdcc_nslots"] = self.rdcc_nslots
        if self.rdcc_w0 is not None:
            kwargs["rdcc_w0"] = self.rdcc_w0
        return kwargs

    def _get_handle(self) -> h5py.File:
        current_pid = os.getpid()
        if self._h5_file is not None and self._h5_pid != current_pid:
            self._h5_file.close()
            self._h5_file = None
            self._h5_pid = None
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", **self._open_kwargs())
            self._h5_pid = current_pid
        return self._h5_file

    def _get_group(self) -> h5py.Group:
        return self._get_handle()[self.split]

    def read(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Read and dequantize one image plus all aligned masks and metadata."""

        if index < 0:
            index += self.n_images
        if index < 0 or index >= self.n_images:
            raise IndexError(index)

        group = self._get_group()
        stored = group["images"][index]
        if int(stored.min()) < 0:
            raise ValueError(f"image {index} contains negative int16 storage values")
        image_np = (
            stored.astype(np.float32) * self.dequantize_scale
            + self.dequantize_offset
        )
        if not np.isfinite(image_np).all():
            raise ValueError(f"image {index} contains NaN or infinity after dequantization")
        if float(image_np.min()) < -1e-6 or float(image_np.max()) > 1.0 + 1e-6:
            raise ValueError(
                f"image {index} dequantizes outside [0,1]: "
                f"[{float(image_np.min())}, {float(image_np.max())}]"
            )

        raw_metadata = group["metadata/json"].asstr()[index]
        metadata = json.loads(raw_metadata)
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata for image {index} must decode to an object")
        if not isinstance(metadata.get("source_id"), str) or not metadata["source_id"]:
            raise ValueError(f"metadata for image {index} requires a non-empty source_id")

        masks = {
            name: torch.from_numpy(group[f"masks/{name}"][index].astype(np.uint8))
            for name in self.mask_names
        }
        return torch.from_numpy(image_np), masks, metadata

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            finally:
                self._h5_file = None
                self._h5_pid = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(h5_file={str(self.h5_path)!r}, "
            f"split={self.split!r}, num_images={self.n_images}, image_shape=(1, "
            f"{self.image_height}, {self.image_width}), masks={self.mask_names}, "
            f"schema_version={self.schema_version!r}, "
            f"dequantization_source={self.dequantization_source!r}, "
            f"rdcc_nbytes={self.rdcc_nbytes!r}, rdcc_nslots={self.rdcc_nslots!r}, "
            f"rdcc_w0={self.rdcc_w0!r})"
        )


_ROTATION_INTERPOLATIONS = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
}

_RESIZE_INTERPOLATIONS = {
    "nearest": TF.InterpolationMode.NEAREST,
    "bilinear": TF.InterpolationMode.BILINEAR,
    "bicubic": TF.InterpolationMode.BICUBIC,
}


def interpolation_mode(name: str, *, operation: str) -> TF.InterpolationMode:
    choices = (
        _ROTATION_INTERPOLATIONS
        if operation == "rotation"
        else _RESIZE_INTERPOLATIONS
    )
    if name not in choices:
        raise ValueError(
            f"unsupported {operation} interpolation {name!r}; "
            f"choose from {sorted(choices)}"
        )
    return choices[name]


def _minimum_padding_free_crop_size(
    image_height: int,
    image_width: int,
    *,
    random_rotation: bool,
    rotation_range: tuple[float, float],
) -> int:
    """Return the guaranteed square crop size over the configured rotations."""

    minimum_dimension = min(int(image_height), int(image_width))
    if not random_rotation:
        return minimum_dimension

    low, high = rotation_range
    if high - low >= 90.0:
        maximum_extent_factor = math.sqrt(2.0)
    else:
        candidate_angles = [low, high]
        first_diagonal = math.ceil((low - 45.0) / 90.0)
        last_diagonal = math.floor((high - 45.0) / 90.0)
        if first_diagonal <= last_diagonal:
            candidate_angles.append(45.0 + 90.0 * first_diagonal)
        maximum_extent_factor = max(
            abs(math.cos(math.radians(angle)))
            + abs(math.sin(math.radians(angle)))
            for angle in candidate_angles
        )
    return int(math.floor(minimum_dimension / maximum_extent_factor))


def _valid_window_origins(valid: torch.Tensor, patch_size: int) -> torch.Tensor:
    if valid.ndim != 2:
        raise ValueError(f"validity mask must be 2-D, got {tuple(valid.shape)}")
    height, width = valid.shape
    if height < patch_size or width < patch_size:
        return torch.empty((0, 2), dtype=torch.int64)

    values = valid.to(torch.int64)
    integral = torch.zeros((height + 1, width + 1), dtype=torch.int64)
    integral[1:, 1:] = values.cumsum(dim=0).cumsum(dim=1)
    window_sums = (
        integral[patch_size:, patch_size:]
        - integral[:-patch_size, patch_size:]
        - integral[patch_size:, :-patch_size]
        + integral[:-patch_size, :-patch_size]
    )
    return torch.nonzero(window_sums == patch_size * patch_size, as_tuple=False)


def rotate_and_valid_crop(
    image: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    angle_deg: float,
    patch_size: int,
    image_interpolation: str,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"image must have shape [1,H,W], got {tuple(image.shape)}")
    image_mode = interpolation_mode(image_interpolation, operation="rotation")
    rotated_image = TF.rotate(
        image, float(angle_deg), interpolation=image_mode, expand=False, fill=0.0
    )
    rotated_support = TF.rotate(
        torch.ones_like(image),
        float(angle_deg),
        interpolation=image_mode,
        expand=False,
        fill=0.0,
    )
    valid = rotated_support[0] >= 1.0 - 1e-6
    origins = _valid_window_origins(valid, patch_size)
    if len(origins) == 0:
        raise ValueError(
            f"rotated image {tuple(image.shape[-2:])} has no padding-free "
            f"{patch_size}x{patch_size} crop at angle {angle_deg:.6f}"
        )
    origin_index = int(rng.integers(0, len(origins)))
    top, left = (int(value) for value in origins[origin_index].tolist())
    clean = rotated_image[:, top : top + patch_size, left : left + patch_size]

    rotated_masks: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        if mask.shape != image.shape:
            raise ValueError(
                f"mask {name!r} has shape {tuple(mask.shape)}, "
                f"expected {tuple(image.shape)}"
            )
        rotated = TF.rotate(
            mask.to(torch.float32),
            float(angle_deg),
            interpolation=TF.InterpolationMode.NEAREST,
            expand=False,
            fill=0.0,
        )
        rotated_masks[name] = rotated[
            :, top : top + patch_size, left : left + patch_size
        ].to(torch.uint8)

    return clean, rotated_masks, {
        "angle_deg": float(angle_deg),
        "rotation_interpolation": image_interpolation,
        "crop_xywh": (left, top, patch_size, patch_size),
        "valid_crop_candidate_count": int(len(origins)),
    }


def rescale_up_and_center_crop(
    image: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    factor: float,
    image_interpolation: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError(f"rescale factor must be finite and >= 1, got {factor}")
    height, width = image.shape[-2:]
    new_height = max(height, int(math.ceil(height * factor)))
    new_width = max(width, int(math.ceil(width * factor)))
    image_mode = interpolation_mode(image_interpolation, operation="resize")

    if new_height == height and new_width == width:
        resized_image = image
        resized_masks = masks
    else:
        resized_image = TF.resize(
            image,
            [new_height, new_width],
            interpolation=image_mode,
            antialias=image_mode
            in {TF.InterpolationMode.BILINEAR, TF.InterpolationMode.BICUBIC},
        )
        resized_masks = {
            name: TF.resize(
                mask,
                [new_height, new_width],
                interpolation=TF.InterpolationMode.NEAREST,
            ).to(torch.uint8)
            for name, mask in masks.items()
        }

    top = (new_height - height) // 2
    left = (new_width - width) // 2
    clean = resized_image[:, top : top + height, left : left + width]
    pre_clamp_min = float(clean.min())
    pre_clamp_max = float(clean.max())
    below_zero = clean < 0.0
    above_one = clean > 1.0
    clean = clean.clamp(0.0, 1.0)
    cropped_masks = {
        name: mask[:, top : top + height, left : left + width]
        for name, mask in resized_masks.items()
    }
    return clean, cropped_masks, {
        "rescale_factor": float(factor),
        "rescale_output_hw": (new_height, new_width),
        "rescale_image_interpolation": image_interpolation,
        "rescale_mask_interpolation": "nearest",
        "rescale_center_crop_xywh": (left, top, width, height),
        "rescale_pre_clamp_min": pre_clamp_min,
        "rescale_pre_clamp_max": pre_clamp_max,
        "rescale_clamp_applied": bool(below_zero.any() or above_one.any()),
        "rescale_clamp_low_pixel_count": int(below_zero.sum()),
        "rescale_clamp_high_pixel_count": int(above_one.sum()),
    }


def _validate_tensor(
    image: torch.Tensor,
    rng: np.random.Generator,
    *,
    name: str,
    require_unit_range: bool,
) -> tuple[float, float]:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if image.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} must contain only finite values")
    minimum = float(image.min())
    maximum = float(image.max())
    if require_unit_range and (minimum < -1e-7 or maximum > 1.0 + 1e-7):
        raise ValueError(f"{name} must be normalized to [0, 1] before noise")
    return minimum, maximum


def _validate_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _validate_nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def add_scan_noise(
    clean: torch.Tensor,
    *,
    jx: float,
    jy: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    jx = _validate_nonnegative_finite(jx, "jx")
    jy = _validate_nonnegative_finite(jy, "jy")
    clean_min, clean_max = _validate_tensor(
        clean, rng, name="clean", require_unit_range=True
    )
    if clean.ndim == 2:
        plane = clean
        restore_channel = False
    elif clean.ndim == 3 and clean.shape[0] == 1:
        plane = clean[0]
        restore_channel = True
    else:
        raise ValueError("clean must have shape [H,W] or [1,H,W]")
    height, width = (int(value) for value in plane.shape)
    if height != width:
        raise ValueError("the historical scan-noise model requires a square image")

    coordinates_1d = range(height)
    x_coordinates, y_coordinates = np.meshgrid(coordinates_1d, coordinates_1d)
    dx = rng.normal(0.0, 1.0, (height, 1)) * jx
    dy = rng.normal(0.0, 1.0, (1, width)) * jy
    coordinates = np.array([y_coordinates + dy, x_coordinates + dx])
    warped_np = map_coordinates(
        plane.detach().to(torch.float32).cpu().numpy(), coordinates
    )
    warped_plane = torch.as_tensor(warped_np, dtype=clean.dtype, device=clean.device)
    warped = warped_plane.unsqueeze(0) if restore_channel else warped_plane

    if jx > 0.0 and jy == 0.0:
        inferred_axis = "x"
    elif jy > 0.0 and jx == 0.0:
        inferred_axis = "y"
    elif jx == 0.0 and jy == 0.0:
        inferred_axis = "none"
    else:
        inferred_axis = "xy"

    return warped, {
        "scan_noise_model": "historical_line_jitter_map_coordinates_v1",
        "scan_axis_inferred": inferred_axis,
        "scan_jitter_x_sigma_px": jx,
        "scan_jitter_y_sigma_px": jy,
        "scan_interpolation_order": 3,
        "scan_boundary_mode": "constant",
        "scan_boundary_cval": 0.0,
        "scan_prefilter": True,
        "pre_scan_min": clean_min,
        "pre_scan_max": clean_max,
        "post_scan_raw_min": float(warped.min()),
        "post_scan_raw_max": float(warped.max()),
        "normalization_status": "not_applied",
    }


def _sample_poisson_peak_sigma(
    clean: torch.Tensor,
    *,
    peak_sigma: float,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    peak_sigma = _validate_positive_finite(peak_sigma, "peak_sigma")
    eps = _validate_positive_finite(eps, "eps")
    _, clean_max = _validate_tensor(
        clean, rng, name="clean", require_unit_range=True
    )
    clean_nonnegative = clean.clamp_min(0.0)
    clean_mean = float(clean_nonnegative.mean())
    gain = 1.0 / (peak_sigma**2)
    expected_counts = clean_nonnegative.detach().to(torch.float64).cpu().numpy() * gain
    sampled_counts = rng.poisson(expected_counts)
    noisy = torch.as_tensor(
        sampled_counts / gain, dtype=clean.dtype, device=clean.device
    )
    realized_rmse = float(torch.sqrt(torch.mean((noisy - clean_nonnegative) ** 2)))
    return noisy, {
        "poisson_parameterization": "peak_sigma_v1",
        "poisson_peak_sigma": peak_sigma,
        "poisson_gain": gain,
        "clean_mean": clean_mean,
        "clean_max": clean_max,
        "expected_mean_counts_per_pixel": clean_mean * gain,
        "expected_peak_counts": clean_max * gain,
        "expected_global_rmse_raw": peak_sigma * np.sqrt(clean_mean),
        "expected_observed_peak_sigma": peak_sigma * np.sqrt(clean_max),
        "realized_global_rmse_raw": realized_rmse,
        "poisson_zero_signal": clean_max <= eps,
    }


def _sample_gaussian(
    image: torch.Tensor, *, sigma: float, rng: np.random.Generator
) -> tuple[torch.Tensor, dict[str, Any]]:
    gaussian = rng.normal(0.0, sigma, size=tuple(image.shape))
    residual = torch.as_tensor(gaussian, dtype=image.dtype, device=image.device)
    noisy = image + residual
    return noisy, {
        "gaussian_sigma": sigma,
        "gaussian_realized_mean": float(residual.mean()),
        "gaussian_realized_std": float(residual.std(unbiased=False)),
        "pre_gaussian_min": float(image.min()),
        "pre_gaussian_max": float(image.max()),
        "post_gaussian_min": float(noisy.min()),
        "post_gaussian_max": float(noisy.max()),
    }


def minmax_normalize_noisy(
    image: torch.Tensor,
    *,
    epsilon: float = float(np.finfo(np.float32).eps),
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")
    if image.numel() == 0:
        raise ValueError("image must not be empty")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("image must contain only finite values")
    epsilon = _validate_positive_finite(epsilon, "epsilon")
    minimum = float(image.min())
    maximum = float(image.max())
    value_range = maximum - minimum
    constant_or_near_constant = value_range <= epsilon
    normalized = torch.zeros_like(image) if constant_or_near_constant else (
        image - minimum
    ) / value_range
    return normalized, {
        "noisy_normalization": "per_image_minmax",
        "normalization_epsilon": epsilon,
        "normalization_constant_or_near_constant": constant_or_near_constant,
        "pre_normalization_min": minimum,
        "pre_normalization_max": maximum,
        "pre_normalization_range": value_range,
        "post_normalization_min": float(normalized.min()),
        "post_normalization_max": float(normalized.max()),
    }


def add_scan_poisson_gaussian_noise(
    clean: torch.Tensor,
    *,
    scan_axis: str,
    scan_jitter_sigma: float,
    poisson_peak_sigma: float,
    gaussian_sigma: float,
    rng: np.random.Generator,
    normalization_epsilon: float = float(np.finfo(np.float32).eps),
) -> tuple[torch.Tensor, dict[str, Any]]:
    scan_axis = str(scan_axis)
    if scan_axis not in {"x", "y"}:
        raise ValueError("scan_axis must be 'x' or 'y'")
    scan_jitter_sigma = _validate_nonnegative_finite(
        scan_jitter_sigma, "scan_jitter_sigma"
    )
    jx = scan_jitter_sigma if scan_axis == "x" else 0.0
    jy = scan_jitter_sigma if scan_axis == "y" else 0.0
    scan_raw, scan_metadata = add_scan_noise(clean, jx=jx, jy=jy, rng=rng)

    scan_clamp_tolerance = 1e-7
    below_zero = scan_raw < -scan_clamp_tolerance
    above_one = scan_raw > 1.0 + scan_clamp_tolerance
    scan_clamped = scan_raw.clamp(0.0, 1.0)
    scan_range_metadata: dict[str, Any] = {
        "scan_axis": scan_axis,
        "scan_jitter_sigma_px": scan_jitter_sigma,
        "scan_pre_poisson_clamp": "[0,1]",
        "scan_clamp_tolerance": scan_clamp_tolerance,
        "scan_clamp_applied": bool(below_zero.any() or above_one.any()),
        "scan_clamp_low_pixel_count": int(below_zero.sum()),
        "scan_clamp_high_pixel_count": int(above_one.sum()),
        "post_scan_clamp_min": float(scan_clamped.min()),
        "post_scan_clamp_max": float(scan_clamped.max()),
    }

    poisson_raw, poisson_metadata = _sample_poisson_peak_sigma(
        scan_clamped, peak_sigma=poisson_peak_sigma, rng=rng
    )
    gaussian_sigma = _validate_positive_finite(gaussian_sigma, "gaussian_sigma")
    _validate_tensor(
        poisson_raw, rng, name="poisson_raw", require_unit_range=False
    )
    mixed_raw, gaussian_metadata = _sample_gaussian(
        poisson_raw, sigma=gaussian_sigma, rng=rng
    )
    noisy, normalization_metadata = minmax_normalize_noisy(
        mixed_raw, epsilon=normalization_epsilon
    )
    metadata: dict[str, Any] = {
        "noise_type": "scan_poisson_gaussian",
        "noise_protocol": "mixed_scan_peak_sigma_linear_v1",
        "noise_order": (
            "scan_then_clamp_then_poisson_then_gaussian_then_per_image_minmax"
        ),
        "clean_normalization": "not_applied_after_geometry",
        **{
            key: value
            for key, value in scan_metadata.items()
            if key != "normalization_status"
        },
        **scan_range_metadata,
        **poisson_metadata,
        **gaussian_metadata,
        **normalization_metadata,
    }
    return noisy, metadata


def collate_denoising_samples(
    samples: list[dict[str, object]],
) -> dict[str, object]:
    """Stack image tensors while preserving heterogeneous metadata records."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    mask_names = tuple(samples[0]["masks"].keys())
    for sample in samples:
        if tuple(sample["masks"].keys()) != mask_names:
            raise ValueError("all samples in a batch must contain the same masks")
    return {
        "noisy": torch.stack([sample["noisy"] for sample in samples]),
        "clean": torch.stack([sample["clean"] for sample in samples]),
        "masks": {
            name: torch.stack([sample["masks"][name] for sample in samples])
            for name in mask_names
        },
        "metadata": [sample["metadata"] for sample in samples],
    }


@dataclass(frozen=True)
class RescalePolicy:
    """Source-specific post-crop upscaling policy."""

    enabled: bool = False
    factor_range: tuple[float, float] = (1.0, 1.5)
    image_interpolation: str = "bilinear"

    def validate(self) -> None:
        if len(self.factor_range) != 2:
            raise ValueError("factor_range must contain exactly two values")
        low, high = (float(value) for value in self.factor_range)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("factor_range must be finite")
        if low < 1.0 or high < low or high > 1.5:
            raise ValueError(
                "T301 rescale factor_range must satisfy 1.0 <= low <= high <= 1.5"
            )


@dataclass(frozen=True)
class MixedNoisePolicy:
    """Scan, Poisson, Gaussian, and normalization policy."""

    scan_jitter_sigma_range: tuple[float, float] = SCAN_JITTER_SIGMA_RANGE
    gaussian_sigma_range: tuple[float, float] = GAUSSIAN_SIGMA_RANGE
    poisson_peak_sigma_range: tuple[float, float] = POISSON_PEAK_SIGMA_RANGE
    scan_axis_sampling: str = "uniform_xy"
    parameter_sampling: str = "linear_uniform"
    normalization: str = "per_image_minmax"
    normalization_epsilon: float = float(np.finfo(np.float32).eps)

    @staticmethod
    def _validate_subrange(
        values: tuple[float, float],
        *,
        name: str,
        allowed: tuple[float, float],
    ) -> None:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        low, high = (float(value) for value in values)
        allowed_low, allowed_high = allowed
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"{name} must be finite")
        if low > high or low < allowed_low or high > allowed_high:
            raise ValueError(
                f"{name} must satisfy {allowed_low} <= low <= high <= {allowed_high}"
            )

    def validate(self) -> None:
        self._validate_subrange(
            self.scan_jitter_sigma_range,
            name="scan_jitter_sigma_range",
            allowed=SCAN_JITTER_SIGMA_RANGE,
        )
        self._validate_subrange(
            self.gaussian_sigma_range,
            name="gaussian_sigma_range",
            allowed=GAUSSIAN_SIGMA_RANGE,
        )
        self._validate_subrange(
            self.poisson_peak_sigma_range,
            name="poisson_peak_sigma_range",
            allowed=POISSON_PEAK_SIGMA_RANGE,
        )
        if self.parameter_sampling != "linear_uniform":
            raise ValueError("parameter_sampling must be 'linear_uniform'")
        if self.scan_axis_sampling != "uniform_xy":
            raise ValueError("scan_axis_sampling must be 'uniform_xy'")
        if self.normalization != "per_image_minmax":
            raise ValueError("normalization must be 'per_image_minmax'")
        if not np.isfinite(self.normalization_epsilon) or self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be finite and > 0")


class DenoisingPatchDataset(Dataset):
    """Load one combined HDF5 and produce aligned padding-free patches.

    Processing order: rotate, valid crop, source-aware upscale, optional flips,
    scan jitter, Poisson noise, Gaussian noise, then noisy-image min-max.
    """

    def __init__(
        self,
        h5_file: str | Path,
        *,
        split: str = "train",
        n_patches: int = 1,
        patch_size: int = 512,
        random_rotation: bool = True,
        rotation_range: tuple[float, float] = (0.0, 360.0),
        rotation_interpolation: str = "bilinear",
        source_rescale_policies: Mapping[str, RescalePolicy] | None = None,
        noise_policy: MixedNoisePolicy = MixedNoisePolicy(),
        horizontal_flip_probability: float = 0.0,
        vertical_flip_probability: float = 0.0,
        seed: int = 42,
        deterministic: bool = True,
        hdf5_rdcc_nbytes: int | None = None,
        hdf5_rdcc_nslots: int | None = None,
        hdf5_rdcc_w0: float | None = None,
    ) -> None:
        self.source = StandardH5Source(
            h5_file,
            split=split,
            rdcc_nbytes=hdf5_rdcc_nbytes,
            rdcc_nslots=hdf5_rdcc_nslots,
            rdcc_w0=hdf5_rdcc_w0,
        )
        self.n_patches = int(n_patches)
        self.patch_size = int(patch_size)
        self.random_rotation = bool(random_rotation)
        self.rotation_range = self._validate_range(rotation_range, "rotation_range")
        self.rotation_interpolation = str(rotation_interpolation)
        self.horizontal_flip_probability = self._validate_probability(
            horizontal_flip_probability, "horizontal_flip_probability"
        )
        self.vertical_flip_probability = self._validate_probability(
            vertical_flip_probability, "vertical_flip_probability"
        )
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        self.deterministic = bool(deterministic)
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()

        if not isinstance(noise_policy, MixedNoisePolicy):
            raise TypeError("noise_policy must be MixedNoisePolicy")
        noise_policy.validate()
        self.noise_policy = noise_policy
        if self.n_patches < 1:
            raise ValueError("n_patches must be >= 1")
        if self.patch_size < 1 or self.patch_size > 512:
            raise ValueError("patch_size must satisfy 1 <= patch_size <= 512")
        guaranteed_crop_size = _minimum_padding_free_crop_size(
            self.source.image_height,
            self.source.image_width,
            random_rotation=self.random_rotation,
            rotation_range=self.rotation_range,
        )
        if self.patch_size > guaranteed_crop_size:
            rotation_description = (
                f"rotation_range={self.rotation_range}"
                if self.random_rotation
                else "rotation disabled"
            )
            raise ValueError(
                f"images with shape ({self.source.image_height}, "
                f"{self.source.image_width}) are too small for "
                f"patch_size={self.patch_size} with {rotation_description}; "
                f"the guaranteed padding-free crop size is "
                f"{guaranteed_crop_size}"
            )

        self.source_rescale_policies = dict(source_rescale_policies or {})
        for source_id, policy in self.source_rescale_policies.items():
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("source_rescale_policies keys must be non-empty strings")
            if not isinstance(policy, RescalePolicy):
                raise TypeError("source_rescale_policies values must be RescalePolicy")
            policy.validate()

    @staticmethod
    def _validate_range(values: tuple[float, float], name: str) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        low, high = (float(value) for value in values)
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            raise ValueError(f"{name} must be finite and satisfy low <= high")
        return low, high

    @staticmethod
    def _validate_probability(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
        return value

    def __len__(self) -> int:
        return len(self.source) * self.n_patches

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self._epoch_state.fill_(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch_state.item())

    def _rng_for_index(
        self, index: int, image_index: int, patch_index: int
    ) -> np.random.Generator:
        if not self.deterministic:
            return np.random.default_rng()
        sequence = np.random.SeedSequence(
            [self.seed, self.epoch, int(index), int(image_index), int(patch_index)]
        )
        return np.random.default_rng(sequence)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        image_index = index // self.n_patches
        patch_index = index % self.n_patches
        image, masks, source_metadata = self.source.read(image_index)
        rng = self._rng_for_index(index, image_index, patch_index)
        angle = (
            float(rng.uniform(*self.rotation_range)) if self.random_rotation else 0.0
        )
        clean, masks, geometry_metadata = rotate_and_valid_crop(
            image,
            masks,
            angle_deg=angle,
            patch_size=self.patch_size,
            image_interpolation=self.rotation_interpolation,
            rng=rng,
        )

        source_id = source_metadata["source_id"]
        policy = self.source_rescale_policies.get(source_id, RescalePolicy())
        factor = float(rng.uniform(*policy.factor_range)) if policy.enabled else 1.0
        clean, masks, rescale_metadata = rescale_up_and_center_crop(
            clean,
            masks,
            factor=factor,
            image_interpolation=policy.image_interpolation,
        )

        horizontal_flip = bool(rng.random() < self.horizontal_flip_probability)
        vertical_flip = bool(rng.random() < self.vertical_flip_probability)
        flip_dims: list[int] = []
        if vertical_flip:
            flip_dims.append(-2)
        if horizontal_flip:
            flip_dims.append(-1)
        if flip_dims:
            clean = torch.flip(clean, dims=flip_dims)
            masks = {
                name: torch.flip(mask, dims=flip_dims) for name, mask in masks.items()
            }

        scan_axis = "x" if int(rng.integers(0, 2)) == 0 else "y"
        scan_jitter_sigma = float(
            rng.uniform(*self.noise_policy.scan_jitter_sigma_range)
        )
        poisson_peak_sigma = float(
            rng.uniform(*self.noise_policy.poisson_peak_sigma_range)
        )
        gaussian_sigma = float(rng.uniform(*self.noise_policy.gaussian_sigma_range))
        noisy, noise_metadata = add_scan_poisson_gaussian_noise(
            clean,
            scan_axis=scan_axis,
            scan_jitter_sigma=scan_jitter_sigma,
            poisson_peak_sigma=poisson_peak_sigma,
            gaussian_sigma=gaussian_sigma,
            rng=rng,
            normalization_epsilon=self.noise_policy.normalization_epsilon,
        )
        seed_components = (
            (self.seed, self.epoch, int(index), int(image_index), int(patch_index))
            if self.deterministic
            else ()
        )
        metadata = {
            **source_metadata,
            "dataset_index": int(index),
            "image_index": int(image_index),
            "patch_index": int(patch_index),
            "epoch": int(self.epoch),
            "seed": int(self.seed),
            "deterministic": self.deterministic,
            **geometry_metadata,
            "source_rescale_override": bool(policy.enabled),
            "rescale_factor_range": tuple(float(v) for v in policy.factor_range),
            **rescale_metadata,
            "horizontal_flip": horizontal_flip,
            "vertical_flip": vertical_flip,
            "noise_parameter_sampling": self.noise_policy.parameter_sampling,
            "scan_axis_sampling": self.noise_policy.scan_axis_sampling,
            "scan_jitter_sigma_range": tuple(
                float(value) for value in self.noise_policy.scan_jitter_sigma_range
            ),
            "gaussian_sigma_range": tuple(
                float(value) for value in self.noise_policy.gaussian_sigma_range
            ),
            "poisson_peak_sigma_range": tuple(
                float(value) for value in self.noise_policy.poisson_peak_sigma_range
            ),
            "noise_seed_components": seed_components,
            **noise_metadata,
            "clean_range": (float(clean.min()), float(clean.max())),
            "noisy_range": (float(noisy.min()), float(noisy.max())),
        }
        return {
            "noisy": noisy.to(torch.float32),
            "clean": clean.to(torch.float32),
            "masks": masks,
            "metadata": metadata,
        }

    def close(self) -> None:
        self.source.close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(h5_file={str(self.source.h5_path)!r}, "
            f"num_images={len(self.source)}, n_patches={self.n_patches}, "
            f"patch_size={self.patch_size}, rotation_range={self.rotation_range}, "
            f"noise_policy={self.noise_policy!r}, seed={self.seed}, "
            f"deterministic={self.deterministic})"
        )


__all__ = [
    "DenoisingPatchDataset",
    "MixedNoisePolicy",
    "RescalePolicy",
    "collate_denoising_samples",
]
