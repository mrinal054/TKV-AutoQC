"""
MONAI ResNet implementation:
    https://docs.monai.io/en/0.9.1/_modules/monai/networks/nets/resnet.html
"""

import torch
import torch.nn as nn
from monai.networks.nets import resnet
from classification_head import ClassificationHeadWithoutFlatten

class ResNetFeatures(nn.Module):
    def __init__(self, model_name="resnet18", in_channels=1, flatten=False):
        super().__init__()
        self.flatten = flatten

        # Instantiate full model
        base = getattr(resnet, model_name)(
            spatial_dims=3, 
            n_input_channels=in_channels, 
        )

        # Manually take only the layers before flatten
        self.features = nn.Sequential(*list(base.children())[:-2])  # conv1 to layer4
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.features(x)  # Avoids avgpool and flatten
        if self.flatten:
            x = self.pool(x).view(x.size(0), -1)
        return x


class ResNetClassifier(nn.Module):
    def __init__(self, model_name="resnet18", 
                 n_classes=15, 
                 in_channels=1,
                 out_channels:list=None, # for instance [1024, 512, 256]. Used in classification head
                 dropout:float=0.3,
                 device="cuda"):
        super().__init__()

        if model_name not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            ValueError("Invalid model_name. Choose from: \
                       resnet18, resnet34, resnet50, resnet101, resnet152")

        # Get ResNet features (Not the flatten version, so that the dummy tensor shape can be calculated)
        self.backbone = ResNetFeatures(
            model_name=model_name,
            in_channels=in_channels,
            flatten=False
        ).to(device)

        # Dynamically calculate no. of output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 96, 96, 96).to(device) # [B, C, D, H, W]
            features = self.backbone(dummy)
            feature_dim = features.shape[1]

        # Global average pooling 
        self.pool = nn.AdaptiveAvgPool3d(1) # [B, C, 1, 1, 1]

        # Classification head
        self.classifier = ClassificationHeadWithoutFlatten(num_classes=n_classes,
                                                 out_channels=[feature_dim] + out_channels,
                                                 dropout=dropout)

        
    def forward(self, x):
        x = self.backbone(x)               # e.g. [B, C, D/32, H/32, W/32]
        x = self.pool(x).view(x.size(0), -1)  # [B, C]
        x = self.classifier(x)            # [B, n_classes]
        return x


#%% Example case
if __name__ == "__main__":
    input = torch.randn(1, 1, 128, 128, 128)
    device = "cuda"
    model = ResNetClassifier(model_name="resnet18", 
                            n_classes=15, 
                            in_channels=1,
                            out_channels=[512, 256, 128],
                            device=device)
    model = model.to(device)
    print(model)
    input = input.to(device)

    output = model(input) 

    print(f"Output shape: {output.shape}")

