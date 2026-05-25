from .mamba_dual_head import MODEL_ALIASES, MambaVisionDualHead
from .resnet_dual_head import ResNetDualHead, create_resnet_dual_head_model

MODEL_CHOICES = ("mamba_t", "mamba_s", "resnet50")


def create_dual_head_model(
    model_name: str = "mamba_t",
    num_weather_classes: int = 4,
    num_object_classes: int = 18,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    freeze_backbone: bool = True,
    pretrained_backbone: bool = True,
):
    if model_name in MODEL_ALIASES or model_name.startswith("nvidia/"):
        from .mamba_dual_head import create_dual_head_model as create_mamba_dual_head_model

        return create_mamba_dual_head_model(
            model_name=model_name,
            num_weather_classes=num_weather_classes,
            num_object_classes=num_object_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    if model_name == "resnet50":
        return create_resnet_dual_head_model(
            num_weather_classes=num_weather_classes,
            num_object_classes=num_object_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            pretrained_backbone=pretrained_backbone,
        )

    raise ValueError(f"Unknown model_name={model_name!r}. Expected one of {MODEL_CHOICES}.")


__all__ = [
    "MODEL_ALIASES",
    "MODEL_CHOICES",
    "MambaVisionDualHead",
    "ResNetDualHead",
    "create_dual_head_model",
]
