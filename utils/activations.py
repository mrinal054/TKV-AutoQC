import torch.nn as nn

# Activation function
def activation_function(name):
    supported_names = ['relu', 'leakyrelu', 'elu', 'gelu', 'sigmoid', 'softmax']
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'leakyrelu':
        activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == 'gelu':
        activation = nn.GELU()
    elif name == 'elu':
        activation = nn.ELU(alpha=1.0, inplace=True)
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    elif name == 'softmax':
        activation = nn.Softmax(dim=1)
    else:
        raise ValueError(f"{name} is not found in supported activation functions: {supported_names}")

    return activation

        