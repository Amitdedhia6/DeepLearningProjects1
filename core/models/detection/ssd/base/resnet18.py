import torch
from collections import OrderedDict
from torch import nn
from torchvision.models import resnet18

#################################################################################
# Base Model
#################################################################################

class Resnet18Base(nn.Module):

    def __init__(self):
        super(Resnet18Base, self).__init__()

        pretrained = resnet18(pretrained=True)
        pretrained_modules = list(pretrained.children())
        # (N, 3, 300, 300)
        self.conv_1 = nn.Sequential(*pretrained_modules[:3])
        # (N, 64, 150, 150)
        self.pool_1 = pretrained_modules[3]
        # (N, 64, 75, 75)
        self.conv_2x = pretrained_modules[4]
        # (N, 64, 75, 75)
        self.conv_3x = pretrained_modules[5]
        # (N, 128, 38, 38) Used as conv_3x feature
        self.conv_4x = pretrained_modules[6]
        # (N, 256, 19, 19) Used as conv_4x feature
        self.conv_5x = pretrained_modules[7]
        # (N, 512, 10, 10) Used as conv_5x feature

    def load_pretrained_weights(self):
        pass

#################################################################################
# Extra layers
#################################################################################

def conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True))


resnet18_extra = nn.Sequential(OrderedDict([
    ("extra_1_1", conv_module(512, 128, kernel_size=1, padding=0)),
    ("extra_1_2", conv_module(128, 256, stride=2)),
    ("extra_2_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_2_2", conv_module(128, 256, stride=2)),
    ("extra_3_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_3_2", conv_module(128, 256, padding=0))
]))

#################################################################################
# Config
#################################################################################

resnet18_config = {
    "base": Resnet18Base,
    "extra": resnet18_extra,
    "n_channels": [128, 256, 512, 256, 256, 256],
    "feature_layers": ["conv_3x", "conv_4x", "conv_5x", "extra_1_2", "extra_2_2", "extra_3_2"]
}
