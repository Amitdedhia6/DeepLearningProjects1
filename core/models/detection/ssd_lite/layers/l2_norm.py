import torch
from torch import nn

class L2Norm(nn.Module):

    def __init__(self, n_channels):
        super(L2Norm, self).__init__()
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, n_channels, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

    def forward(self, x):
        # L2 Normalization
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        output = torch.div(x, norm)
        output = output * self.rescale_factors
        return output
