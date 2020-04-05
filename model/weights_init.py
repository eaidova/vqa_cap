import torch.nn as nn


def weights_init_xn(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)


def weights_init_xu(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)


def weights_init_ku(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data, a=0.01)


WEIGHTS_INIT = {
    'xavier_normal': weights_init_xn,
    'xavier_uniform': weights_init_xu,
    'kaiming_normal': weights_init_kn,
    'kaiming_uniform': weights_init_ku
}


def init_weights(model, init):
    initializer = WEIGHTS_INIT.get(init)
    if initializer:
        model.apply(initializer)