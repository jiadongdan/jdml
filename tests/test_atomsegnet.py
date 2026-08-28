import torch

from jdml.models import AtomSegNetNestedUNet, AtomSegNetUNet


def test_atomsegnet_unet_output_shape_and_range():
    model = AtomSegNetUNet().eval()

    with torch.no_grad():
        output = model(torch.randn(1, 1, 16, 20))

    assert output.shape == (1, 1, 16, 20)
    assert torch.all((0 <= output) & (output <= 1))


def test_atomsegnet_unet_supports_multiple_channels():
    model = AtomSegNetUNet(channels=2).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 2, 8, 12))

    assert output.shape == (1, 2, 8, 12)


def test_atomsegnet_nested_unet_output_shape_and_range():
    model = AtomSegNetNestedUNet(filters=(2, 4, 8, 16, 32)).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 1, 16, 32))

    assert output.shape == (1, 1, 16, 32)
    assert torch.all((-1 <= output) & (output <= 1))


def test_atomsegnet_models_are_publicly_exported():
    from jdml import models

    assert models.AtomSegNetUNet is AtomSegNetUNet
    assert models.AtomSegNetNestedUNet is AtomSegNetNestedUNet
