def model_params(name, config=None):
    """ Register model parameters here """
    
    # Create a dictionary to store model parameters
    param = dict()
    
    if name == "ResNetClassifier":
        param["model_name"] = config.model.subname
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        param["device"] = config.train.device
    elif name == "DenseNetClassifier":
        param["model_name"] = config.model.subname
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        param["device"] = config.train.device
    elif name == "ImageLatentsClassifier":
        param["return_latents_only"] = config.model.return_latents_only
        param["n_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["freeze_latents"] = config.model.freeze_latents
        param["dropout"] = config.model.dropout
        param["load_ckpt"] = config.model.load_ckpt
    elif name == "ImageLatentsClassifierV2":
        param["return_latents_only"] = config.model.return_latents_only
        param["n_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["freeze_latents"] = config.model.freeze_latents
        param["dropout"] = config.model.dropout
        param["load_ckpt"] = config.model.load_ckpt        
    elif name == "VGG16Classifier": # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["out_channels"] = config.model.out_channels or []
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["device"] = config.train.device
    elif name == "ViTClassifier": # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["img_size"] = tuple(config.model.img_size)
        param["patch_size"] = tuple(config.model.patch_size)
        param["hidden_size"] = config.model.hidden_size
        param["mlp_dim"] = config.model.mlp_dim
        param["num_layers"] = config.model.num_layers
        param["num_heads"] = config.model.num_heads
        param["pos_embed_type"] = config.model.pos_embed_type
        param["spatial_dims"] = getattr(config.model, "spatial_dims", 3)
    elif name == "SwinUNETRClassifier":  # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["attn_drop_rate"] = getattr(config.model, "attn_drop_rate", 0.0)
        param["feature_size"] = getattr(config.model, "feature_size", 48)
        param["use_checkpoint"] = getattr(config.model, "use_checkpoint", False)
        param["spatial_dims"] = getattr(config.model, "spatial_dims", 3)
    elif name == "EfficientNetClassifier":  # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["model_name"] = config.model.subname  # e.g., efficientnet-b0, b1
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["spatial_dims"] = getattr(config.model, "spatial_dims", 3)
        param["pretrained_path"] = getattr(config.model, "pretrained_path", None)  # path to .pth weights
    elif name == "MedicalNetResNet18Classifier":  # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["out_channels"] = config.model.out_channels or []
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["device"] = config.train.device
        param["pretrained_path"] = getattr(config.model, "pretrained_path", None)  # path to .pth weights
        param["input_shape"] = getattr(config.model, "input_shape", (64, 128, 128))  # your MRI volumes
        param["freeze_backbone"] = getattr(config.model, "freeze_backbone", None)    
    elif name == "MultiImageNet":
        param["config_path"] = config.model.config_path
        param["n_classes"] = config.train.n_classes
        param["out_channels"] = config.model.out_channels
        param["dropout"] = config.model.dropout
        param["device"] = config.train.device
        param["dummy_size"] = config.model.dummy_size # [1, 1, 96, 96, 96] # for 3D
        param["verbose"] = False

    else:
        raise ValueError(f"{name} is not found in supported model list")

    return param
