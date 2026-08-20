import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRFeatures(nn.Module):
    """
    MultiImageNet-compatible SwinUNETR backbone.
    Returns [B, C] feature vectors.
    """
    def __init__(
        self,
        in_channels=1,
        img_size=(64, 128, 128),
        feature_size=48,
        use_checkpoint=False,
        dropout=0.0,
        attn_drop_rate=0.0,
        spatial_dims=3,
        device="cuda"
    ):
        super().__init__()

        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=1,          # unused
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            drop_rate=dropout,
            attn_drop_rate=attn_drop_rate
        )

        # Infer final embedding dim safely (device-consistent)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *img_size)
            dummy = dummy.to(next(self.backbone.parameters()).device)

            hidden_states = self.backbone.swinViT(dummy)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            feat_dim = hidden_states[-1].shape[1]

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_dim = feat_dim
        self.to(device)

    def forward(self, x):
        hidden_states = self.backbone.swinViT(x)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        x = hidden_states[-1]          # [B, C, D', H', W']
        x = self.pool(x).flatten(1)    # [B, C]
        return x


class SwinUNETRClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_size=(64, 128, 128),
        feature_size=48,  # controls embedding dim
        use_checkpoint=False,
        dropout=0.0,
        attn_drop_rate=0.0,
        n_classes=1,
        spatial_dims=3
    ):
        super().__init__()

        self.backbone = SwinUNETR(
            in_channels=in_channels,
            out_channels=n_classes,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            drop_rate=dropout,
            attn_drop_rate=attn_drop_rate
        )

        # Get last stage feature dim dynamically (device-consistent)
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            dummy_input = dummy_input.to(next(self.backbone.parameters()).device)

            hidden_states = self.backbone.swinViT(dummy_input)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            feat_dim = hidden_states[-1].shape[1]

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        hidden_states = self.backbone.swinViT(x)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        x = hidden_states[-1]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
