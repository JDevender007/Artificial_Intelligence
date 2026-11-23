"""Models: hand-written linear/MLP layers, a small CNN, and a pretrained ResNet-18.

``ScratchLinear`` implements the affine layer and He initialisation directly with
``nn.Parameter`` so the MLP contains no ``nn.Linear``; a unit test checks it matches
``torch.nn.functional.linear`` exactly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ScratchLinear(nn.Module):
    """``y = x W^T + b`` with He (Kaiming-normal) init, written out by hand."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # He init keeps activation variance ~constant through ReLU layers: Var(w) = 2 / fan_in.
        with torch.no_grad():
            self.weight.normal_(0.0, math.sqrt(2.0 / self.in_features))
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.t()
        return out + self.bias if self.bias is not None else out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class LinearClassifier(nn.Module):
    """Softmax regression: the simplest baseline (one ScratchLinear on raw pixels)."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = ScratchLinear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(x, 1))


class MLP(nn.Module):
    """Fully connected net: [ScratchLinear -> BatchNorm -> ReLU -> Dropout] x N -> logits."""

    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (512, 256),
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [ScratchLinear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        self.body = nn.ModuleList(layers)
        self.dropout = dropout
        self.head = ScratchLinear(prev, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        for layer in self.body:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = F.dropout(x, self.dropout, self.training)
        return self.head(x)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 conv-BN-ReLU layers then 2x2 max-pool (VGG-style)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Three conv blocks (64-128-256) + global average pooling + dropout + linear head (~1.15M params)."""

    def __init__(self, in_channels: int, n_classes: int, widths: tuple[int, ...] = (64, 128, 256),
                 dropout: float = 0.3) -> None:
        super().__init__()
        blocks, prev = [], in_channels
        for w in widths:
            blocks.append(_block(prev, w))
            prev = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(prev, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        return self.head(self.drop(torch.flatten(x, 1)))


def resnet18(n_classes: int, pretrained: bool = True) -> nn.Module:
    """torchvision ResNet-18 with a new ``n_classes`` head.

    ``weights=`` replaced the removed ``pretrained=True`` argument (torchvision >= 0.13).
    Weights download once to ``$TORCH_HOME`` (here D:\\Python_Environment\\cache\\torch).
    """
    from torchvision.models import ResNet18_Weights
    from torchvision.models import resnet18 as tv_resnet18

    model = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def build_model(name: str, in_channels: int, img_size: int, n_classes: int,
                pretrained: bool = True) -> nn.Module:
    """Factory for every model in the comparison."""
    in_dim = in_channels * img_size * img_size
    if name == "linear":
        return LinearClassifier(in_dim, n_classes)
    if name == "mlp":
        return MLP(in_dim, n_classes)
    if name == "cnn":
        return SmallCNN(in_channels, n_classes)
    if name == "resnet18":
        return resnet18(n_classes, pretrained)
    raise ValueError(f"Unknown model {name!r}")
