"""
MedicalNet ResNet18 classifier wrapper for 3D volumes
"""

import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from classification_head import ClassificationHeadWithoutFlatten

# Adjust the path if necessary so Python can find your MedicalNet models folder
sys.path.append("/research/m324371/Project/Digital_Twin/Classification/MedicalNet/")  
from MedicalNet.models import resnet  # This should point to your MedicalNet resnet.py

class MedicalNetResNet18Features(nn.Module):
    def __init__(self, in_channels=1, pretrained_path=None, device="cuda"):
        super().__init__()
        self.device = device

        # Instantiate MedicalNet ResNet18
        self.backbone = resnet.resnet18(
            sample_input_D=64,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=1,  # we don't care about segmentation here
            shortcut_type='B',
            no_cuda=False
        )

        # Load pretrained weights if provided
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=device)
            self.backbone.load_state_dict(state_dict, strict=False)

        # Take only feature extractor layers (all layers before conv_seg)
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        self.pool = nn.AdaptiveAvgPool3d(1)  # global avg pool

    def forward(self, x, flatten=True):
        x = self.features(x)
        if flatten:
            x = self.pool(x).view(x.size(0), -1)
        return x
    
class MedicalNetResNet18FeaturesV2(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 pretrained_path=None, 
                 flatten:bool=False,
                 device="cuda"):
        super().__init__()
        self.flatten = flatten
        self.device = device

        # Instantiate MedicalNet ResNet18
        self.backbone = resnet.resnet18(
            sample_input_D=64,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=1,  # we don't care about segmentation here
            shortcut_type='B',
            no_cuda=False
        )

        # Load pretrained weights if provided
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location=device)
            self.backbone.load_state_dict(state_dict, strict=False)

        # Take only feature extractor layers (all layers before conv_seg)
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
        if self.flatten: self.pool = nn.AdaptiveAvgPool3d(1)  # global avg pool

    def forward(self, x):
        x = self.features(x)
        if self.flatten:
            x = self.pool(x).view(x.size(0), -1)
        return x


class MedicalNetResNet18Classifier(nn.Module):
    def __init__(self, 
                 n_classes=1, 
                 in_channels=1, 
                 out_channels=None, 
                 dropout=0.3,
                 pretrained_path=None,
                 device="cuda",
                 input_shape=None,
                 freeze_backbone=False):
        super().__init__()
        out_channels = out_channels or []

        self.device = device

        # Backbone
        self.backbone = MedicalNetResNet18Features(
            in_channels=in_channels,
            pretrained_path=pretrained_path,
            device=device
        ).to(device)

        if freeze_backbone:
            # default: freeze everything
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 128, 128).to(device)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]

        # Pooling (already handled in backbone, but keeping for consistency)
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Classification head
        self.classifier = ClassificationHeadWithoutFlatten(
            num_classes=n_classes,
            out_channels=[feature_dim] + out_channels,
            dropout=dropout
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, C]
        x = self.classifier(x)
        return x


# %% Example usage
if __name__ == "__main__":
    device = "cuda"
    model = MedicalNetResNet18Classifier(
        n_classes=1,
        in_channels=1,
        out_channels=[512, 256],
        dropout=0.3,
        pretrained_path=None,
        device=device
    ).to(device)

    input_tensor = torch.randn(1, 1, 64, 128, 128).to(device)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
