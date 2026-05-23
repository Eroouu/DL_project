from __future__ import annotations

import torch
from torch import nn


MODEL_ALIASES = {
    "mamba_t": {"hf_id": "nvidia/MambaVision-T-1K", "feature_dim": 640},
    "mamba_s": {"hf_id": "nvidia/MambaVision-S-1K", "feature_dim": 640},
}


class MambaVisionDualHead(nn.Module):
    def __init__(
        self,
        model_name: str = "mamba_t",
        num_weather_classes: int = 4,
        num_object_classes: int = 19,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_weather_classes = num_weather_classes
        self.num_object_classes = num_object_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        resolved_name, feature_dim = self._resolve_model(model_name)
        self.resolved_model_name = resolved_name
        self.feature_dim = feature_dim

        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(resolved_name, trust_remote_code=True)

        self.weather_head = self._make_head(num_weather_classes)
        self.object_head = self._make_head(num_object_classes)
        self.set_backbone_trainable(not freeze_backbone)

    @staticmethod
    def _resolve_model(model_name: str) -> tuple[str, int]:
        if model_name in MODEL_ALIASES:
            spec = MODEL_ALIASES[model_name]
            return spec["hf_id"], spec["feature_dim"]
        return model_name, 640

    def _make_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
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

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(images)

        if torch.is_tensor(outputs):
            features = outputs
        elif isinstance(outputs, tuple):
            features = outputs[0]
        elif isinstance(outputs, dict):
            features = None
            for key in ("pooler_output", "last_hidden_state", "logits"):
                if key in outputs and outputs[key] is not None:
                    features = outputs[key]
                    break
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state
        elif hasattr(outputs, "logits"):
            features = outputs.logits
        else:
            raise TypeError(f"Unsupported backbone output type: {type(outputs)}")

        if features is None or not torch.is_tensor(features):
            raise TypeError(f"Backbone did not return tensor features: {type(outputs)}")
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim != 2:
            features = features.flatten(start_dim=1)
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.backbone_trainable:
            features = self.extract_features(images)
        else:
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                features = self.extract_features(images)
            self.backbone.train(was_training)

        return {
            "weather_logits": self.weather_head(features),
            "object_logits": self.object_head(features),
        }


def create_dual_head_model(
    model_name: str = "mamba_t",
    num_weather_classes: int = 4,
    num_object_classes: int = 19,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
) -> MambaVisionDualHead:
    return MambaVisionDualHead(
        model_name=model_name,
        num_weather_classes=num_weather_classes,
        num_object_classes=num_object_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
