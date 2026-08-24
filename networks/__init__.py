"""Public model registry for TKV-AutoQC."""

from networks.resnet import ResNetClassifier, ResNetFeatures
from networks.densenet import DenseNetClassifier, DenseNetFeatures
from networks.multiImageNet import MultiImageNet
from networks.vgg16 import VGG16Classifier, VGG16Features3DV2
from networks.vit import ViTClassifier, ViTFeatures
from networks.swin_unetr import SwinUNETRClassifier, SwinUNETRFeatures
from networks.efficientnet import EffNetClassifier, EffNetFeatures


__all__ = [
    "ResNetClassifier",
    "ResNetFeatures",
    "DenseNetClassifier",
    "DenseNetFeatures",
    "MultiImageNet",
    "VGG16Classifier",
    "VGG16Features3DV2",
    "ViTClassifier",
    "ViTFeatures",
    "SwinUNETRClassifier",
    "SwinUNETRFeatures",
    "EffNetClassifier",
    "EffNetFeatures",
]
