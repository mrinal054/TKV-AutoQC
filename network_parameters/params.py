def model_params(name, config=None):
    """ Register model parameters here """
    
    # Create a dictionary to store model parameters
    param = dict()

    if name == "MedicalNetResNet18Classifier":  # added by Abby
        param["n_classes"] = config.train.n_classes
        param["in_channels"] = config.model.in_channels
        param["out_channels"] = config.model.out_channels or []
        param["dropout"] = config.model.dropout if config.model.dropout is not None else 0.0
        param["device"] = config.train.device
        param["pretrained_path"] = getattr(config.model, "pretrained_path", None)  # path to .pth weights
        param["input_shape"] = getattr(config.model, "input_shape", (64, 128, 128))  # MRI volumes
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
