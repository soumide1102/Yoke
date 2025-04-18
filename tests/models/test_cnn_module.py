"""Test CNNmodule."""

import pytest
import torch
from torch import nn

from yoke.models.CNNmodules import CNN_Interpretability_Module
from yoke.models.CNNmodules import CNN_Reduction_Module
from yoke.models.CNNmodules import PVI_SingleField_CNN


###############################################################################
# Fixtures for CNN_Interpretability_Module
###############################################################################
@pytest.fixture
def default_interpretability_model() -> CNN_Interpretability_Module:
    """Pytest fixture for creating a default interpretability model."""
    return CNN_Interpretability_Module()


###############################################################################
# Tests for CNN_Interpretability_Module
###############################################################################
def test_default_forward_shape(
    default_interpretability_model: CNN_Interpretability_Module,
) -> None:
    """Test that the default interpretability model outputs the expected shape."""
    batch_size = 2
    c_in, height, width = default_interpretability_model.img_size
    x = torch.randn(batch_size, c_in, height, width)
    out = default_interpretability_model(x)
    assert out.shape == (
        batch_size,
        default_interpretability_model.features,
        height,
        width,
    )


def test_custom_model_forward_shape() -> None:
    """Test that a custom-configuration."""
    model = CNN_Interpretability_Module(
        img_size=(3, 224, 224),
        kernel=3,
        features=16,
        depth=4,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 16, 224, 224)


def test_batchnorm_weights_frozen_interpretability() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias is True."""
    model = CNN_Interpretability_Module(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_batchnorm_weights_trainable_interpretability() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias is False."""
    model = CNN_Interpretability_Module(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_conv_bias_toggle_interpretability() -> None:
    """Test that the convolutional bias toggles correctly for interpretability."""
    model_no_bias = CNN_Interpretability_Module(conv_onlyweights=True)
    model_with_bias = CNN_Interpretability_Module(conv_onlyweights=False)
    assert model_no_bias.inConv.bias is None
    assert model_with_bias.inConv.bias is not None


def test_forward_pass_no_exceptions_interpretability(
    default_interpretability_model: CNN_Interpretability_Module,
) -> None:
    """Test that the forward pass does not raise exceptions."""
    x = torch.randn(1, *default_interpretability_model.img_size)
    _ = default_interpretability_model(x)


def test_parameter_count_interpretability() -> None:
    """Test that parameter count for interpretability model is > 0."""
    model = CNN_Interpretability_Module()
    params = list(model.parameters())
    assert len(params) > 0


###############################################################################
# Fixtures for CNN_Reduction_Module
###############################################################################
@pytest.fixture
def default_reduction_model() -> CNN_Reduction_Module:
    """Pytest fixture for creating a default reduction model."""
    return CNN_Reduction_Module()


###############################################################################
# Tests for CNN_Reduction_Module
###############################################################################
def test_default_reduction_forward_shape(
    default_reduction_model: CNN_Reduction_Module,
) -> None:
    """Test that the default reduction model outputs a smaller (or equal) shape."""
    batch_size = 2
    c_in, h_in, w_in = default_reduction_model.img_size
    x = torch.randn(batch_size, c_in, h_in, w_in)
    out = default_reduction_model(x)
    assert out.shape[0] == batch_size
    assert out.shape[1] == default_reduction_model.features
    # Height/width should match finalH/finalW
    assert out.shape[2] == default_reduction_model.finalH
    assert out.shape[3] == default_reduction_model.finalW


def test_custom_reduction_forward_shape() -> None:
    """Test custom-configured reduction model."""
    model = CNN_Reduction_Module(
        img_size=(3, 128, 128),
        size_threshold=(16, 16),
        kernel=3,
        stride=2,
        features=8,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
    )
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    # Check channel count
    assert out.shape[1] == 8
    # Ensure final shape is <= (16, 16)
    assert out.shape[2] <= 16
    assert out.shape[3] <= 16


def test_batchnorm_weights_frozen_reduction() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias is True."""
    model = CNN_Reduction_Module(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_batchnorm_weights_trainable_reduction() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias is False."""
    model = CNN_Reduction_Module(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_conv_bias_toggle_reduction() -> None:
    """Test that the convolutional bias toggles correctly for reduction module."""
    model_no_bias = CNN_Reduction_Module(conv_onlyweights=True)
    model_with_bias = CNN_Reduction_Module(conv_onlyweights=False)
    assert model_no_bias.inConv.bias is None
    assert model_with_bias.inConv.bias is not None


def test_forward_pass_no_exceptions_reduction(
    default_reduction_model: CNN_Reduction_Module,
) -> None:
    """Test forward pass of the reduction model."""
    x = torch.randn(1, *default_reduction_model.img_size)
    _ = default_reduction_model(x)


def test_parameter_count_reduction() -> None:
    """Test that parameter count for the reduction model is > 0."""
    model = CNN_Reduction_Module()
    params = list(model.parameters())
    assert len(params) > 0


###############################################################################
# Fixtures for PVI_SingleField_CNN
###############################################################################
@pytest.fixture
def default_pvi_model() -> PVI_SingleField_CNN:
    """Fixture for creating a default PVI_SingleField_CNN."""
    return PVI_SingleField_CNN()


###############################################################################
# Tests for PVI_SingleField_CNN
###############################################################################
def test_default_pvi_forward_shape(default_pvi_model: PVI_SingleField_CNN) -> None:
    """Test that the default PVI model produces a scalar output (batch, 1)."""
    batch_size = 2
    c_in, height, width = default_pvi_model.img_size
    x = torch.randn(batch_size, c_in, height, width)
    out = default_pvi_model(x)
    # Should be [batch_size, 1]
    assert out.shape == (batch_size, 1)


def test_custom_pvi_forward_shape() -> None:
    """Test that a custom-configured PVI model produces a scalar output."""
    model = PVI_SingleField_CNN(
        img_size=(3, 224, 224),
        size_threshold=(16, 16),
        kernel=3,
        features=8,
        interp_depth=3,
        conv_onlyweights=False,
        batchnorm_onlybias=False,
        act_layer=nn.ReLU,
        hidden_features=10,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)


def test_pvi_batchnorm_weights_frozen() -> None:
    """Test that batchnorm weights are frozen if batchnorm_onlybias=True."""
    model = PVI_SingleField_CNN(batchnorm_onlybias=True)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert not param.requires_grad


def test_pvi_batchnorm_weights_trainable() -> None:
    """Test that batchnorm weights are trainable if batchnorm_onlybias=False."""
    model = PVI_SingleField_CNN(batchnorm_onlybias=False)
    for name, param in model.named_parameters():
        if "Norm" in name and "weight" in name:
            assert param.requires_grad


def test_pvi_conv_bias_toggle() -> None:
    """Test that the convolutional bias toggles correctly."""
    model_no_bias = PVI_SingleField_CNN(conv_onlyweights=True)
    model_with_bias = PVI_SingleField_CNN(conv_onlyweights=False)
    # Check the first convolution in the interpretability module
    assert model_no_bias.interp_module.inConv.bias is None
    assert model_with_bias.interp_module.inConv.bias is not None


def test_pvi_forward_pass_no_exceptions(default_pvi_model: PVI_SingleField_CNN) -> None:
    """Test that the forward pass does not raise exceptions for PVI model."""
    x = torch.randn(1, *default_pvi_model.img_size)
    _ = default_pvi_model(x)


def test_parameter_count_pvi() -> None:
    """Test that parameter count for the PVI model is > 0."""
    model = PVI_SingleField_CNN()
    params = list(model.parameters())
    assert len(params) > 0
