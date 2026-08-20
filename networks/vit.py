import torch
import torch.nn as nn
from monai.networks.nets import ViT

class ViTFeatures(nn.Module):
    """
    MultiImageNet-compatible ViT backbone.
    Returns [B, hidden_size] feature vectors.
    """
    def __init__(
        self,
        in_channels=1,
        img_size=(64, 128, 128),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        pos_embed_type="sincos",
        dropout=0.1,
        spatial_dims=3,
        device="cuda",
        pool="cls",  # "cls" or "mean"
    ):
        super().__init__()

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout,
            classification=False,   # ?? critical
            spatial_dims=spatial_dims,
        )

        assert pool in ["cls", "mean"]
        self.pool = pool
        self.out_dim = hidden_size

        self.to(device)

    def forward(self, x):
        x = self.vit(x)

        # MONAI may return (tokens, hidden_states)
        if isinstance(x, tuple):
            x = x[0]

        # x: [B, N_tokens, hidden_size]
        if self.pool == "cls":
            x = x[:, 0]            # CLS token
        else:
            x = x.mean(dim=1)      # mean pooling

        return x                  # [B, hidden_size]


class ViTClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_size=(64, 128, 128),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        pos_embed_type="perceptron",
        dropout=0.1,
        n_classes=3,
        classification=True,
        spatial_dims=3,
    ):
        super().__init__()

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout,
            classification=classification,
            num_classes=n_classes,
            spatial_dims=spatial_dims,
        )

    def forward(self, x):
            out = self.vit(x)
            if isinstance(out, tuple):
                return out[0]  # discard attention maps, keep logits
            return out