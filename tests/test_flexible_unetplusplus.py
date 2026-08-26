import pytest
import torch

from jdml.models import UNetPlusPlusModel


@pytest.mark.parametrize("up_mode", ["transpose", "bilinear"])
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_output_matches_odd_input_shape(up_mode, kernel_size):
    model = UNetPlusPlusModel(
        in_channels=2,
        out_channels=3,
        features=(4, 8, 16),
        normalization="group",
        up_mode=up_mode,
        kernel_size=kernel_size,
    )

    output = model(torch.randn(1, 2, 17, 23))

    assert output.shape == (1, 3, 17, 23)


def test_backward_reaches_all_parameters_without_deep_supervision():
    model = UNetPlusPlusModel(
        features=(4, 8),
        normalization="none",
    )

    output = model(torch.randn(2, 1, 9, 13))
    output.mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_deep_supervision_returns_one_prediction_per_decoder_stage():
    model = UNetPlusPlusModel(
        in_channels=2,
        out_channels=3,
        features=(4, 8, 16),
        normalization="none",
        deep_supervision=True,
    )

    outputs = model(torch.randn(2, 2, 17, 23))

    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert all(output.shape == (2, 3, 17, 23) for output in outputs)


def test_deep_supervision_backward_reaches_all_parameters():
    model = UNetPlusPlusModel(
        features=(4, 8, 16),
        normalization="none",
        deep_supervision=True,
    )

    outputs = model(torch.randn(2, 1, 17, 19))
    sum(output.mean() for output in outputs).backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_config_round_trip_preserves_model_options():
    model = UNetPlusPlusModel(
        in_channels=2,
        out_channels=3,
        features=(4, 8),
        activation="silu",
        normalization="none",
        dropout=0.1,
        up_mode="bilinear",
        output_activation="softmax",
        kernel_size=5,
        deep_supervision=True,
    )

    rebuilt = UNetPlusPlusModel.from_config(model.get_config())

    assert rebuilt.get_config() == model.get_config()


def test_softmax_requires_multiple_output_channels():
    with pytest.raises(ValueError, match="softmax.*out_channels >= 2"):
        UNetPlusPlusModel(out_channels=1, output_activation="softmax")


def test_deep_supervision_must_be_boolean():
    with pytest.raises(ValueError, match="deep_supervision"):
        UNetPlusPlusModel(deep_supervision=1)


def test_small_input_is_rejected_before_pooling():
    model = UNetPlusPlusModel(
        features=(4, 8, 16),
        normalization="none",
    )

    with pytest.raises(ValueError, match="at least 8"):
        model(torch.randn(1, 1, 7, 20))


def test_batch_norm_rejects_single_bottleneck_value_during_training():
    model = UNetPlusPlusModel(features=(4, 8), normalization="batch")

    with pytest.raises(ValueError, match="Batch normalization"):
        model(torch.randn(1, 1, 4, 4))
