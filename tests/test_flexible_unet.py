import pytest
import torch

from jdml.models import UNetModel


@pytest.mark.parametrize("up_mode", ["transpose", "bilinear"])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_output_matches_odd_input_shape(up_mode, kernel_size):
    model = UNetModel(
        in_channels=2,
        out_channels=3,
        features=(4, 8),
        normalization="group",
        up_mode=up_mode,
        kernel_size=kernel_size,
    )

    output = model(torch.randn(1, 2, 17, 23))

    assert output.shape == (1, 3, 17, 23)


def test_config_round_trip_preserves_model_options():
    model = UNetModel(
        in_channels=2,
        out_channels=3,
        features=(4, 8),
        activation="silu",
        normalization="none",
        dropout=0.1,
        up_mode="bilinear",
        output_activation="softmax",
        kernel_size=5,
    )

    rebuilt = UNetModel.from_config(model.get_config())

    assert rebuilt.get_config() == model.get_config()


def test_softmax_requires_multiple_output_channels():
    with pytest.raises(ValueError, match="softmax.*out_channels >= 2"):
        UNetModel(out_channels=1, output_activation="softmax")


def test_multichannel_softmax_is_normalized():
    model = UNetModel(
        out_channels=3,
        features=(4,),
        normalization="none",
        output_activation="softmax",
    )

    output = model(torch.randn(1, 1, 7, 9))

    torch.testing.assert_close(output.sum(dim=1), torch.ones(1, 7, 9))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": (8.5, 16)}, "features"),
        ({"in_channels": 1.5}, "positive integers"),
        ({"up_mode": None}, "up_mode"),
        ({"activation": None}, "activation"),
        ({"dropout": True}, "dropout"),
    ],
)
def test_invalid_configuration_types_raise_clear_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        UNetModel(**kwargs)


def test_batch_norm_rejects_single_bottleneck_value_during_training():
    model = UNetModel(features=(4, 8), normalization="batch")

    with pytest.raises(ValueError, match="Batch normalization"):
        model(torch.randn(1, 1, 4, 4))


def test_batch_norm_allows_single_bottleneck_value_during_evaluation():
    model = UNetModel(features=(4, 8), normalization="batch").eval()

    output = model(torch.randn(1, 1, 4, 4))

    assert output.shape == (1, 1, 4, 4)


def test_instance_norm_rejects_single_bottleneck_spatial_element():
    model = UNetModel(features=(4, 8), normalization="instance")

    with pytest.raises(ValueError, match="Instance normalization"):
        model(torch.randn(2, 1, 4, 4))


def test_group_norm_rejects_single_value_per_group():
    model = UNetModel(
        features=(1,),
        normalization="group",
        group_norm_groups=8,
    )

    with pytest.raises(ValueError, match="Group normalization"):
        model(torch.randn(1, 1, 2, 2))
