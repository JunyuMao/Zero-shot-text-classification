import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.LSTM):
        init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(2.0))
        init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(2.0))
        # init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(2.0))
        # init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(2.0))
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)