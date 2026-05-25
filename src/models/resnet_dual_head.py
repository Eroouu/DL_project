from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetDualHead(nn.Module):
    def __init__(
        self,
        num_weather_classes: int = 4,
        num_object_classes: int = 18,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = "resnet50"
        self.num_weather_classes = num_weather_classes
        self.num_object_classes = num_object_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.weather_head = self._make_head(num_weather_classes)
        self.object_head = self._make_head(num_object_classes)
        self.set_backbone_trainable(not freeze_backbone)

    def _make_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    @property
    def backbone_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.backbone.parameters())

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def freeze_backbone(self) -> None:
        self.set_backbone_trainable(False)

    def unfreeze_backbone(self) -> None:
        self.set_backbone_trainable(True)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.backbone_trainable:
            features = self.backbone(images)
        else:
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(images)
            self.backbone.train(was_training)

        return {
            "weather_logits": self.weather_head(features),
            "object_logits": self.object_head(features),
        }


def create_resnet_dual_head_model(
    num_weather_classes: int = 4,
    num_object_classes: int = 18,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    pretrained_backbone: bool = True,
) -> ResNetDualHead:
    return ResNetDualHead(
        num_weather_classes=num_weather_classes,
        num_object_classes=num_object_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        pretrained_backbone=pretrained_backbone,
    )
