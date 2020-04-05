import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

NORMALIZATION = {
    'weight': weight_norm,
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
    'none': lambda x, dim: x
}

ACTIVATION = {
    'ReLU':  nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
    'Tanh': nn.Tanh,
    'Hardtanh': nn.Hardtanh,
    'Sigmoid': nn.Sigmoid
}


def get_norm(norm):
    norm_layer = NORMALIZATION.get(norm)
    if not norm_layer:
        raise Exception("Invalid Normalization: {}".format(norm))
    return norm_layer


def get_act(act):
    act_layer = ACTIVATION.get(act)
    if not act_layer:
        raise Exception("Invalid activation function: {}".format(act))
    return act_layer



class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, dropout, norm, act):
        super(FCNet, self).__init__()

        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(norm_layer(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(act_layer())
            layers.append(nn.Dropout(p=dropout))
        layers.append(norm_layer(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(act_layer())
        layers.append(nn.Dropout(p=dropout))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class GTH(nn.Module):
    """Simple class for Gated Tanh
    """
    def __init__(self, in_dim, out_dim, dropout, norm, act):
        super(GTH, self).__init__()

        self.nonlinear = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= act)
        self.gate = FCNet([in_dim, out_dim], dropout= dropout, norm= norm, act= 'Sigmoid')

    def forward(self, x):
        x_proj = self.nonlinear(x)
        gate = self.gate(x)
        x_proj = x_proj*gate
        return x_proj
