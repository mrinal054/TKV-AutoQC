import torch.nn as nn

def loss_func(name, weight=None, *args):
    if name == 'ce':
        if weight == None: return nn.CrossEntropyLoss()
        else: return nn.CrossEntropyLoss(weight=weight)
    elif name == 'bce':
        if weight == None: return nn.BCEWithLogitsLoss()
        else: return nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        raise ValueError(f"{name} is not found in supported losses.")
