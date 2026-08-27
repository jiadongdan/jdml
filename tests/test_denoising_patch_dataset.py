import json

import h5py
import numpy as np
import pytest
import torch

from jdml.dataset._denoising_patch_dataset import (
    DenoisingPatchDataset,
    _minimum_padding_free_crop_size,
    rescale_up_and_center_crop,
)


def _write_standard_h5(path, *, height, width):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "atomic-stem-denoise-h5-v1"
        handle.attrs["dequantize_scale"] = 1.0 / 32767.0
        handle.attrs["dequantize_offset"] = 0.0
        for split in ("train", "valid", "test"):
            group = handle.create_group(split)
            group.create_dataset(
                "images", data=np.zeros((1, 1, height, width), dtype=np.int16)
            )
            metadata = group.create_group("metadata")
            metadata.create_dataset(
                "json",
                data=[json.dumps({"source_id": "test"})],
                dtype=string_dtype,
            )


def test_768_images_support_512_patches_over_full_rotation_range():
    assert _minimum_padding_free_crop_size(
        768,
        768,
        random_rotation=True,
        rotation_range=(0.0, 360.0),
    ) == 543


def test_dataset_rejects_patch_larger_than_guaranteed_rotated_crop(tmp_path):
    h5_path = tmp_path / "small.h5"
    _write_standard_h5(h5_path, height=32, width=32)

    with pytest.raises(ValueError, match="guaranteed padding-free crop size is 22"):
        DenoisingPatchDataset(h5_path, patch_size=32, random_rotation=True)


def test_dataset_accepts_full_image_patch_when_rotation_is_disabled(tmp_path):
    h5_path = tmp_path / "small.h5"
    _write_standard_h5(h5_path, height=32, width=32)

    dataset = DenoisingPatchDataset(
        h5_path,
        patch_size=32,
        random_rotation=False,
    )
    try:
        assert len(dataset) == 1
    finally:
        dataset.close()


def test_bicubic_rescale_clamps_interpolation_overshoot():
    image = torch.zeros((1, 8, 8), dtype=torch.float32)
    image[:, :, 4:] = 1.0

    clean, _, metadata = rescale_up_and_center_crop(
        image,
        {},
        factor=1.5,
        image_interpolation="bicubic",
    )

    assert float(clean.min()) == 0.0
    assert float(clean.max()) == 1.0
    assert metadata["rescale_clamp_applied"] is True
    assert metadata["rescale_clamp_low_pixel_count"] > 0
    assert metadata["rescale_clamp_high_pixel_count"] > 0


def test_dataset_rejects_negative_seed(tmp_path):
    h5_path = tmp_path / "data.h5"
    _write_standard_h5(h5_path, height=32, width=32)

    with pytest.raises(ValueError, match="seed must be >= 0"):
        DenoisingPatchDataset(
            h5_path,
            patch_size=16,
            random_rotation=False,
            seed=-1,
        )


def test_dataset_rejects_negative_epoch(tmp_path):
    h5_path = tmp_path / "data.h5"
    _write_standard_h5(h5_path, height=32, width=32)
    dataset = DenoisingPatchDataset(
        h5_path,
        patch_size=16,
        random_rotation=False,
    )
    try:
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            dataset.set_epoch(-1)
    finally:
        dataset.close()
