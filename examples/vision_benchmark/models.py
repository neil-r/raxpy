"""Model architectures for vision benchmark experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer


class ViTWrapper(nn.Module):
    """ViT-Small wrapper supporting single-channel inputs and dynamic patch padding."""

    def __init__(
        self,
        patch_size: int = 4,
        num_layers: int = 12,
        num_heads: int = 6,
        hidden_dim: int = 384,
        mlp_dim: int = 1536,
        dropout: float = 0.0,
        num_classes: int = 10,
        base_img_size: int = 28,
    ):
        super().__init__()
        self.patch_size = patch_size
        remainder = base_img_size % patch_size
        self.pad = (patch_size - remainder) % patch_size
        self.img_size = base_img_size + self.pad

        self.vit = VisionTransformer(
            image_size=self.img_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_classes=num_classes,
        )
        self.vit.conv_proj = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (0, self.pad, 0, self.pad))
        return self.vit(x)


def build_resnet34(
    stem_kernel: int = 3,
    dropout_rate: float = 0.0,
    num_classes: int = 10,
) -> nn.Module:
    """Instantiate a ResNet-34 backbone adapted for single-channel image classification."""
    model = models.resnet34(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=stem_kernel,
        stride=1,
        padding=stem_kernel // 2,
        bias=False,
    )
    model.maxpool = nn.Identity()
    if dropout_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes),
        )
    return model


def build_vit_small(
    patch_size: int = 4,
    dropout_rate: float = 0.0,
    num_classes: int = 10,
) -> nn.Module:
    """Instantiate a ViT-Small (~21.3M parameters) matching ResNet-34 parameter scale."""
    return ViTWrapper(
        patch_size=patch_size,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
        dropout=dropout_rate,
        num_classes=num_classes,
        base_img_size=28,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

