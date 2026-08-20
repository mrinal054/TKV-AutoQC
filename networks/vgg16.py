import torch
import torch.nn as nn
from classification_head import ClassificationHeadWithoutFlatten

class VGG16Features3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        def conv_block(in_c, out_c, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm3d(out_c))  # ? Added BatchNorm
                layers.append(nn.ReLU(inplace=True))
                in_c = out_c
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # Reduced complexity implementation (everything halved)
        #self.features = nn.Sequential(
         #   conv_block(in_channels, 32, 2),
          #  conv_block(32, 64, 2),
           # conv_block(64, 128, 3),
            #conv_block(128, 256, 3),
            #conv_block(256, 256, 3)
        #)

        
        # Regular implementation
        self.features = nn.Sequential(
            conv_block(in_channels, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
           # conv_block(512, 512, 3)
        )

    def forward(self, x):
        return self.features(x)

class VGG16Features3DV2(nn.Module):
    """
    VGG16-style 3D backbone compatible with MultiImageNet.
    Returns features of shape [B, C].
    """
    def __init__(self, in_channels=1, device="cuda"):
        super().__init__()

        self.backbone = VGG16Features3D(in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.backbone(x)                 # [B, C, D, H, W]
        x = self.pool(x).flatten(1)          # [B, C]
        return x

class VGG16Classifier(nn.Module):
    def __init__(self,
                 in_channels=1,
                 n_classes=15,
                 out_channels=None,
                 dropout=None,
                 device="cuda"):
        super().__init__()
        
        if dropout is None:
            dropout = 0.0

        self.backbone = VGG16Features3D(in_channels=in_channels).to(device)

        # Dynamically compute output channels
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 96, 96, 96).to(device)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = ClassificationHeadWithoutFlatten(
            num_classes=n_classes,
            out_channels=[feature_dim] + (out_channels or []),
            dropout=dropout
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = VGG16Classifier(n_classes=3, in_channels=1, out_channels=[512, 256], device="cuda")
    x = torch.randn(1, 1, 128, 128, 128).to("cuda")
    y = model(x)
    print("Output shape:", y.shape)  # Should be [1, 3] for 3-class classification
