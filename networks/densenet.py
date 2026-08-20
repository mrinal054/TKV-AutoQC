"""
MONAI DenseNet implementation:
https://docs.monai.io/en/stable/networks.html#densenet
"""
# import sys
# sys.path.append("/research/m324371/Project/Digital_Twin/Classification/utils/")

import torch
import torch.nn as nn
from monai.networks.nets import densenet
from classification_head import ClassificationHeadWithoutFlatten

class DenseNetFeatures(nn.Module):
    def __init__(self, model_name="DenseNet121", in_channels=1, flatten=False):
        super().__init__()
        self.flatten = flatten

        # Instantiate the full model
        self.base = getattr(densenet, model_name)(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1  # placeholder, will remove classifier later
        )

        # Remove the final classifier
        self.base.class_layers = nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.base.features(x)
        if self.flatten:
            x = self.pool(x).view(x.size(0), -1)
        return x


class DenseNetClassifier(nn.Module):
    def __init__(self, 
                 model_name="DenseNet121", 
                 n_classes=15, 
                 in_channels=1, 
                 out_channels: list=None, # for instance [1024, 512, 256]. Used in classification head
                 dropout: float=0.3,
                 device="cuda"):
        super().__init__()

        if model_name not in ["DenseNet121", "DenseNet169", "DenseNet201", "DenseNet264"]:
            ValueError("Invalid model_name. Choose from: \
                       DenseNet121, DenseNet169, DenseNet201, DenseNet264")

        # Get DenseNet feature extractor
        self.backbone = DenseNetFeatures(model_name=model_name, 
                                         in_channels=in_channels, 
                                         flatten=False).to(device)

        # Dynamically calculate output feature size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 96, 96, 96).to(device)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]

        # Global average pooling 
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Classification head
        self.classifier = ClassificationHeadWithoutFlatten(num_classes=n_classes,
                                                 out_channels=[feature_dim] + out_channels,
                                                 dropout=dropout)


    def forward(self, x):
        x = self.backbone(x)                # [B, C, D', H', W']
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)              # [B, n_classes]
        return x


#%% Example usage
if __name__ == "__main__":
    input = torch.randn(1, 1, 128, 128, 128).to("cuda")
    model = DenseNetClassifier(model_name="DenseNet121", 
                            n_classes=15, 
                            in_channels=1,
                            out_channels=[512, 256, 128],
                            dropout=0.3,
                            device="cuda").to("cuda")
    output = model(input)
    print(f"Output shape: {output.shape}")  # should be [1, 15]

    print(model)