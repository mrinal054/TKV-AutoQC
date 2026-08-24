import torch.nn as nn

def loss_func(name, weight=None, label_smoothing=0.0, *args):
    if name == "ce":
        if weight is None:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
    elif name == 'bce':
        if weight == None: return nn.BCEWithLogitsLoss()
        else: return nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        raise ValueError(f"{name} is not found in supported losses.")
