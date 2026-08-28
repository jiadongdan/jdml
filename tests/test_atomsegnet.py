import pytest
import torch

from jdml.models import AtomSegNetNestedUNet, AtomSegNetUNet


def test_atomsegnet_unet_output_shape_and_range():
    model = AtomSegNetUNet().eval()

    with torch.no_grad():
        output = model(torch.randn(1, 1, 16, 20))

    assert output.shape == (1, 1, 16, 20)
    assert torch.all((0 <= output) & (output <= 1))


def test_atomsegnet_unet_supports_multiple_channels():
    model = AtomSegNetUNet(in_channels=2, out_channels=3).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 2, 8, 12))

    assert output.shape == (1, 3, 8, 12)


def test_atomsegnet_nested_unet_output_shape_and_range():
    model = AtomSegNetNestedUNet(
        in_channels=2,
        out_channels=3,
        features=(2, 4, 8, 16, 32),
    ).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 2, 16, 32))

    assert output.shape == (1, 3, 16, 32)
    assert torch.all((-1 <= output) & (output <= 1))


@pytest.mark.parametrize(
    "model",
    [
        AtomSegNetUNet(in_channels=2, out_channels=3),
        AtomSegNetNestedUNet(
            in_channels=2,
            out_channels=3,
            features=(2, 4, 8, 16, 32),
        ),
    ],
)
def test_config_round_trip_preserves_options(model):
    rebuilt = type(model).from_config(model.get_config())

    assert rebuilt.get_config() == model.get_config()


def test_default_architectures_keep_upstream_state_dict_structure():
    unet = AtomSegNetUNet()
    nested_unet = AtomSegNetNestedUNet()

    assert sum(parameter.numel() for parameter in unet.parameters()) == 1_742_533
    assert len(unet.state_dict()) == 106
    assert sum(parameter.numel() for parameter in nested_unet.parameters()) == 9_162_753
    assert len(nested_unet.state_dict()) == 212


@pytest.mark.parametrize(
    ("model", "input_tensor", "message"),
    [
        (AtomSegNetUNet(), torch.randn(1, 2, 8, 8), "input channels"),
        (AtomSegNetUNet(), torch.randn(1, 1, 7, 8), "divisible by 4"),
        (
            AtomSegNetNestedUNet(features=(2, 4, 8, 16, 32)),
            torch.randn(1, 1, 16, 17),
            "divisible by 16",
        ),
    ],
)
def test_invalid_inputs_raise_clear_errors(model, input_tensor, message):
    with pytest.raises(ValueError, match=message):
        model(input_tensor)


def test_legacy_constructor_keywords_remain_available():
    with pytest.warns(DeprecationWarning, match="channels"):
        unet = AtomSegNetUNet(channels=2)
    with pytest.warns(DeprecationWarning, match="filters"):
        nested_unet = AtomSegNetNestedUNet(filters=(2, 4, 8, 16, 32))

    assert unet.get_config() == {"in_channels": 2, "out_channels": 2}
    assert nested_unet.features_config == (2, 4, 8, 16, 32)


def test_atomsegnet_models_are_publicly_exported():
    from jdml import models

    assert models.AtomSegNetUNet is AtomSegNetUNet
    assert models.AtomSegNetNestedUNet is AtomSegNetNestedUNet
