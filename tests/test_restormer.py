import pytest
import torch

from jdml.models import RestormerModel


def _small_model(**kwargs):
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "dim": 4,
        "num_blocks": (1, 1, 1, 1),
        "num_refinement_blocks": 1,
        "heads": (1, 2, 4, 8),
    }
    config.update(kwargs)
    return RestormerModel(**config)


@pytest.mark.parametrize("shape", [(2, 1, 16, 24), (1, 1, 17, 23), (1, 1, 1, 5)])
@pytest.mark.parametrize("layer_norm_type", ["with_bias", "bias_free"])
def test_output_matches_input_shape(shape, layer_norm_type):
    model = _small_model(layer_norm_type=layer_norm_type)

    output = model(torch.randn(*shape))

    assert output.shape == shape


def test_backward_reaches_all_parameters():
    model = _small_model()

    output = model(torch.randn(2, 1, 9, 13))
    output.mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())


def test_zeroed_model_is_identity_with_residual_output():
    model = _small_model()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    image = torch.randn(1, 1, 11, 15)

    output = model(image)

    torch.testing.assert_close(output, image)


def test_different_output_channels_work_without_residual():
    model = _small_model(out_channels=2, residual=False)

    output = model(torch.randn(1, 1, 9, 13))

    assert output.shape == (1, 2, 9, 13)


def test_config_round_trip_preserves_all_options():
    model = _small_model(
        layer_norm_type="BiasFree",
        bias=True,
        residual=True,
        pad_mode="replicate",
    )

    rebuilt = RestormerModel.from_config(model.get_config())

    assert rebuilt.get_config() == model.get_config()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dim": 3}, "positive even"),
        ({"num_blocks": (1, 1, 1)}, "four positive"),
        ({"heads": (1, 3, 4, 8)}, "must be divisible"),
        ({"ffn_expansion_factor": 0.1}, "hidden channel"),
        ({"out_channels": 2}, "residual=True"),
        ({"layer_norm_type": None}, "layer_norm_type"),
        ({"pad_mode": "circular"}, "pad_mode"),
    ],
)
def test_invalid_configuration_raises_clear_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _small_model(**kwargs)


def test_rejects_invalid_input_layout_channel_and_dtype():
    model = _small_model()

    with pytest.raises(ValueError, match="4D"):
        model(torch.randn(1, 8, 8))
    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(1, 2, 8, 8))
    with pytest.raises(ValueError, match="floating-point"):
        model(torch.ones(1, 1, 8, 8, dtype=torch.int64))
