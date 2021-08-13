import torch
from collections import OrderedDict
from torch import nn
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual

#################################################################################
# Base Model
#################################################################################

class MobileNetV2Base(nn.Module):

    def __init__(self):
        super(MobileNetV2Base, self).__init__()

        pretrained = mobilenet_v2(pretrained=True)
        pretrained_modules = list(pretrained.children())[0]
        # (N, 3, 300, 300)
        self.block_13 = nn.Sequential(*pretrained_modules[:14])
        self.block_14_0 = pretrained_modules[14].conv[0]
        self.block_14_1 = pretrained_modules[14].conv[1]
        self.block_14_2 = pretrained_modules[14].conv[2]
        self.block_14_3 = pretrained_modules[14].conv[3]
        self.block_18 = nn.Sequential(*pretrained_modules[15:19])

    def load_pretrained_weights(self):
        pass

#################################################################################
# Extra layers
#################################################################################
mobilenetv2_extra = nn.Sequential(OrderedDict([
    ("extra_1", InvertedResidual(1280, 512, stride=2, expand_ratio=1)),
    ("extra_2", InvertedResidual(512, 256, stride=2, expand_ratio=1)),
    ("extra_3", InvertedResidual(256, 256, stride=2, expand_ratio=1)),
    ("extra_4", InvertedResidual(256, 64, stride=2, expand_ratio=1))
]))

#################################################################################
# Config
#################################################################################

mobilenetv2_config = {
    "base": MobileNetV2Base,
    "extra": mobilenetv2_extra,
    "n_channels": [576, 1280, 512, 256, 256, 64],
    "feature_layers": ["block_14_0", "block_18", "extra_1", "extra_2", "extra_3", "extra_4"]
}
