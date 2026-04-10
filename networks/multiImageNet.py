"""
m324371 @ 13 November 2025
"""
import yaml
import torch
import torch.nn as nn
from classification_head import ClassificationHeadWithoutFlatten

class MultiImageNet(nn.Module):
    def __init__(self,
                 config_path: str = "config_multiImageNet.yaml",
                 n_classes: int = None,
                 out_channels: list = None,
                 dropout: float = None,
                 device: str = None,
                 dummy_size: list = [1, 1, 96, 96, 96], # for 2D: [B, C, H, W]; for 3D: [B, C, D, H, W]
                 verbose: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        
        """
        MultiImageNet combines multiple deep learning backbones that each produce
        a feature vector of shape [B, C]. The feature vectors from all models are
        concatenated along the channel dimension and passed through a classification
        head for final prediction.
        
        :param config_path: (str) Full yaml path specifying model definitions.
            Each entry should include:
                {
                    "ModelKey": {
                        "modelclass": <str: name of model class in `nets`>,
                        "inputs": <dict: keyword arguments passed to model class>
                    },
                    ...
                }
        
        :param n_classes: (int) Number of output classes for the classification head.
        
        :param out_channels: (list) List of hidden layer sizes for the classification
            head. The first element will automatically be prefixed with the total
            concatenated feature dimension.
        
        :param dropout: (float) Dropout probability applied between layers in the
            classification head.
        
        :param device: (str, optional) Device identifier for model initialization
            and computation. Defaults to `"cuda"` if available, otherwise `"cpu"`.
        
        :param dummy_size: (list) Shape of dummy input tensor used to infer feature
            dimensions from each model during initialization.
            - For 2D backbones: [B, C, H, W]
            - For 3D backbones: [B, C, D, H, W]
            Default: [1, 1, 96, 96, 96].
        
        :param verbose: (bool, optional) If True, prints feature dimensions and model
            loading information during initialization. Defaults to False.
        
        :param **kwargs: Additional keyword arguments passed to the base `nn.Module`
            or reserved for future configuration extensions.
        """

        # from . import nets # Lazy import to avoid circular import with networks.nets
        import networks

        # Read the config file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        
        # Backbones are expected to have output size of B x C. So, they should be already averaged
        # pooled and flattened (self.pool(x).view(x.size(0), -1))
        
        self.backbones = nn.ModuleList()
        self.verbose = verbose
        
        # Setup device
        self.device = (
            torch.device(device) if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize all models and calculate their feature dimensions
        feature_dims = []
        for model_key, model_cfg in config.items():
            model_classname = model_cfg["modelclass"]
            model_kwargs = model_cfg["inputs"]
            
            model_class = getattr(networks, model_classname)
            backbone = model_class(**model_kwargs).to(self.device)
            self.backbones.append(backbone) # append model
            
            # Calculate feature dimensions for all models using dummy inputs
            with torch.no_grad():
                dummy = torch.zeros(*dummy_size, device=self.device) 
                features = backbone(dummy)
                feature_dim = features.shape[1] # channel size
                feature_dims.append(feature_dim) 
                
                if self.verbose: print(f"No. of channels in {model_cfg['modelclass']}: {feature_dim}")
        
        total_features = sum(feature_dims)

        # Combine both channels
        if out_channels is None: 
            final_channels = [total_features] 
        else:
            final_channels = [total_features] + out_channels
        
        # Classification head
        self.classifier = ClassificationHeadWithoutFlatten(num_classes=n_classes,
                                                 out_channels=final_channels, 
                                                 dropout=dropout).to(self.device)            

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable or disable gradient updates for all backbone networks."""
        for backbone in self.backbones:
            for p in backbone.parameters():
                p.requires_grad = trainable

    def backbone_parameters(self):
        """Return an iterator over backbone parameters."""
        return (p for backbone in self.backbones for p in backbone.parameters())
    
    def classifier_parameters(self):
        """Return an iterator over classifier parameters."""
        return self.classifier.parameters()
    
    def forward(self, multi_images):
        # multi_images should be either a tuple or list (e.g. [image, mask]).
        
        # Assert no. of  elements in x and self.backbones are the same.
        if isinstance(multi_images, (list, tuple)):
            # Multi-image case: one tensor per backbone
            assert len(multi_images) == len(self.backbones), \
                f"No. of tensor data ({len(multi_images)}) should be equal to no. of models ({len(self.backbones)})."
        elif isinstance(multi_images, torch.Tensor):
            # Single-image case: one tensor for all samples, one backbone only
            assert len(self.backbones) == 1, \
                f"No. of tensor data (1) should be equal to no. of models ({len(self.backbones)})."
            # Wrap in a list so later code can iterate uniformly
            multi_images = [multi_images]
        else:
            raise TypeError(
                f"multi_images must be a Tensor or list/tuple of Tensors, got {type(multi_images)}"
            )
    
        # Loop over data and corresponding model. Store model outputs.
        featurelist = []
        for backbone, data in zip(self.backbones, multi_images):
            data = data.to(self.device)
            feats = backbone(data) # model output [B, C_i]
            featurelist.append(feats) # append model output
            
        # Concatenate features along the channel dimension
        features = torch.cat(featurelist, dim=1) # [B, sum C_i]       
        
        # Pass the features through the classification head
        out = self.classifier(features)

        return out            
            
            
""" Example usage
# To perform a test run with this model, run the following code 
# from the base directory (/research/m324371/Project/Digital_Twin/Classification/)

import os
import networks
import torch
import torch.nn as nn
           
if __name__ == "__main__":
    
    # # Structure of a config file
    # config = {
    #     "Backbone1": {
    #         "modelclass": "ResNetFeatures",
    #         "inputs": {
    #             "model_name": "resnet18",
    #             "in_channels": 1,
    #             "flatten": True
    #         }
    #     },
    #     "Backbone2": {
    #         "modelclass": "ResNetFeatures",
    #         "inputs": {
    #             "model_name": "resnet50",
    #             "in_channels": 1,
    #             "flatten": True
    #         }
    #     }
    # }

    # Two random 3D images corresponding to the two models
    x1 = torch.randn(2, 1, 96, 96, 96)
    x2 = torch.randn(2, 1, 96, 96, 96)   

    MultiImageNet = getattr(networks, "MultiImageNet") 

    model = MultiImageNet(
        config_path="networks/config_multiImageNet.yaml",
        n_classes=3,
        out_channels=[32, 16],
        dropout=0.1,
        device="cpu",
        verbose=True,
    )
    
    out = model([x1, x2])
    
    print(out.shape)

# # It should show the following output:
# No. of channels in ResNetFeatures: 512
# No. of channels in ResNetFeatures: 2048
# torch.Size([2,3])

"""

