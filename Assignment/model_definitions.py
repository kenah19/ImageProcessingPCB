"""Model architectures and checkpoint loading for the PCB inspection app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models


class LightweightViT(nn.Module):
    """Lightweight Vision Transformer used in the training notebook."""

    def __init__(
        self,
        num_classes: int = 6,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])


def _resnet(num_classes: int, dropout_p: float) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p), nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def _mobilenet(num_classes: int, dropout_p: float) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[2].p = dropout_p
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def _alexnet(num_classes: int, dropout_p: float) -> nn.Module:
    model = models.alexnet(weights=None)
    model.classifier[2].p = dropout_p
    model.classifier[5].p = dropout_p
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _model_specific_params(model_name: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "HPO-Tuned" not in model_name:
        return {}

    params = checkpoint.get("hpo_best_params") or {}
    if not isinstance(params, dict):
        return {}

    key_by_family = {
        "LPViT": "LPViT (Custom ViT)",
        "CS-ResNet": "CS-ResNet (ResNet18)",
        "MobileNetV3": "MobileNetV3 (Small)",
        "AlexNet": "AlexNet (Auto-VRS)",
    }
    for family, key in key_by_family.items():
        if family in model_name:
            nested = params.get(key)
            return nested if isinstance(nested, dict) else params
    return {}


def build_model(model_name: str, num_classes: int, params: dict[str, Any]) -> nn.Module:
    """Recreate one of the four architectures exactly as trained."""
    if "LPViT" in model_name:
        return LightweightViT(
            num_classes=num_classes,
            num_heads=int(params.get("num_heads", 3)),
            mlp_ratio=float(params.get("mlp_ratio", 2.0)),
        )
    if "CS-ResNet" in model_name:
        return _resnet(num_classes, float(params.get("dropout_p", 0.0)))
    if "MobileNetV3" in model_name:
        return _mobilenet(num_classes, float(params.get("dropout_p", 0.2)))
    if "AlexNet" in model_name:
        return _alexnet(num_classes, float(params.get("dropout_p", 0.5)))
    raise ValueError(f"Unsupported model: {model_name}")


def load_exported_model(checkpoint_path: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    """Load a portable checkpoint exported by the notebook."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location="cpu")

    model_name = str(checkpoint["model_name"])
    num_classes = int(checkpoint["num_classes"])
    params = _model_specific_params(model_name, checkpoint)
    model = build_model(model_name, num_classes, params)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
