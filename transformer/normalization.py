import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """ Layer Normalization https://arxiv.org/abs/1607.06450 """
    # reference implementation: 
    # https://github.com/CyberZHG/torch-layer-normalization/blob/89f405b60f53f85da6f03fe685c190ef394ce50c/torch_layer_normalization/layer_normalization.py

    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.eps = eps
    
    def forward(self, x):
        # difference from batch norm: x.mean(0) vs x.mean(-1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
