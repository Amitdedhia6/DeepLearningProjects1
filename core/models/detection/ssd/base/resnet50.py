import torch
from collections import OrderedDict
from torch import nn
from torchvision.models import resnet50

#################################################################################
# Base Model
#################################################################################

class Resnet50Base(nn.Module):

    def __init__(self):
        super(Resnet50Base, self).__init__()

        pretrained = resnet50(pretrained=True)
        pretrained_modules = list(pretrained.children())
        # (N, 3, 300, 300)
        self.conv_1 = nn.Sequential(*pretrained_modules[:3])
        # (N, 64, 150, 150)
        self.pool_1 = pretrained_modules[3]
        # (N, 64, 75, 75)
        self.conv_2x = pretrained_modules[4]
        # (N, 256, 75, 75)
        self.conv_3x = pretrained_modules[5]
        # (N, 512, 38, 38)
        # Modify conv 4x so that is does not downsize input
        self.conv_4x = pretrained_modules[6]
        self.conv_4x[0].conv1.stride=(1, 1)
        self.conv_4x[0].conv2.stride=(1, 1)
        self.conv_4x[0].downsample[0].stride=(1, 1)
        # (N, 1024, 38, 38) Used as conv_4x feature

    def load_pretrained_weights(self):
        pass

#################################################################################
# Extra layers
#################################################################################

def conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True))


resnet50_extra = nn.Sequential(OrderedDict([
    ("extra_1_1", conv_module(1024, 256, kernel_size=1, padding=0)),
    ("extra_1_2", conv_module(256, 512, kernel_size=3, padding=1, stride=2)),
    ("extra_2_1", conv_module(512, 256, kernel_size=1, padding=0)),
    ("extra_2_2", conv_module(256, 512, kernel_size=3, padding=1, stride=2)),
    ("extra_3_1", conv_module(512, 128, kernel_size=1, padding=0)),
    ("extra_3_2", conv_module(128, 256, kernel_size=3, padding=1, stride=2)),
    ("extra_4_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_4_2", conv_module(128, 256, kernel_size=3, padding=0)),
    ("extra_5_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_5_2", conv_module(128, 256, kernel_size=3, padding=0))
]))

#################################################################################
# Config
#################################################################################

resnet50_config = {
    "base": Resnet50Base,
    "extra": resnet50_extra,
    "n_channels": [1024, 512, 512, 256, 256, 256],
    "feature_layers": ["conv_4x", "extra_1_2", "extra_2_2", "extra_3_2", "extra_4_2", "extra_5_2"]
}
