import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN

class EffNetFeatures(nn.Module):
    """
    MONAI EfficientNet backbone for MultiImageNet.
    Returns [B, C] pooled features.
    """
    def __init__(self, model_name="efficientnet-b0", in_channels=1, pretrained_path=None):
        super().__init__()

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # Handle both raw state_dict or full checkpoint
            state_dict = checkpoint.get("state_dict", checkpoint)

            missing, unexpected = self.load_state_dict(state_dict, strict=False)

            print(f"[INFO] Loaded pretrained weights from {pretrained_path}")
            if missing:
                print(f"[INFO] Missing keys: {missing}")
            if unexpected:
                print(f"[INFO] Unexpected keys: {unexpected}")
                
        self.base = EfficientNetBN(
            model_name=model_name,
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=1  # placeholder
        )

        # Remove classifier
        self.base._fc = nn.Identity()

        # Global pooling (MONAI still uses this internally)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        # Replicate EfficientNetBN.forward up to pooling
        x = self.base._conv_stem(x)
        x = self.base._bn0(x)
        x = self.base._blocks(x)
        x = self.base._conv_head(x)
        x = self.base._bn1(x)

        x = self.pool(x)
        x = x.flatten(1)
        return x

class EffNetClassifier(nn.Module):
    def __init__(self, model_name="efficientnet-b0", in_channels=1, n_classes=2, dropout=0.0, spatial_dims=3):
        super().__init__()

        # Load 3D EfficientNet from MONAI
        self.backbone = EfficientNetBN(
            model_name=model_name,
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=n_classes
        )

        # MONAI’s EfficientNetBN already includes a classifier head
        # If you want custom dropout:
        if dropout > 0:
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, n_classes)
            )

    def forward(self, x):
        return self.backbone(x)