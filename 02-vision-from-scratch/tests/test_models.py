import pytest
import torch
import torch.nn.functional as F

from dlfund.models import MLP, ScratchLinear, SmallCNN, build_model


def test_scratch_linear_matches_functional_linear():
    torch.manual_seed(0)
    layer = ScratchLinear(7, 3)
    x = torch.randn(5, 7)
    with torch.no_grad():
        layer.bias.normal_()
    assert torch.allclose(layer(x), F.linear(x, layer.weight, layer.bias), atol=1e-6)


def test_scratch_linear_he_init_variance():
    torch.manual_seed(0)
    layer = ScratchLinear(1000, 1000)
    assert abs(layer.weight.var().item() - 2 / 1000) < 2e-4
    assert torch.count_nonzero(layer.bias) == 0


def test_mlp_has_no_torch_linear():
    model = MLP(784, 10)
    assert not any(isinstance(m, torch.nn.Linear) for m in model.modules())


@pytest.mark.parametrize("name,ch,size", [("linear", 1, 28), ("mlp", 1, 28), ("cnn", 1, 28),
                                          ("linear", 3, 32), ("mlp", 3, 32), ("cnn", 3, 32)])
def test_output_shapes(name, ch, size):
    model = build_model(name, ch, size, 10).eval()
    assert model(torch.randn(4, ch, size, size)).shape == (4, 10)


def test_resnet_head_replaced_without_download():
    model = build_model("resnet18", 3, 64, 10, pretrained=False).eval()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 10)


def test_cnn_param_count_is_small():
    n = sum(p.numel() for p in SmallCNN(3, 10).parameters())
    assert 1.0e6 < n < 1.3e6


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("vit", 3, 32, 10)
