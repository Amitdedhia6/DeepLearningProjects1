import torch
from ..layers import L2Norm
from collections import OrderedDict
from torch import nn
from torchvision.models import vgg16_bn

#################################################################################
# Base Model
#################################################################################

class VGG16Base(nn.Module):

    def __init__(self):
        super(VGG16Base, self).__init__()
        # (N, 3, 300, 300)
        self.conv_1_1 = VGG16Base.conv_module(3, 64)
        self.conv_1_2 = VGG16Base.conv_module(64, 64)
        self.pool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 64, 150, 150)
        self.conv_2_1 = VGG16Base.conv_module(64, 128)
        self.conv_2_2 = VGG16Base.conv_module(128, 128)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 128, 75, 75)
        self.conv_3_1 = VGG16Base.conv_module(128, 256)
        self.conv_3_2 = VGG16Base.conv_module(256, 256)
        self.conv_3_3 = VGG16Base.conv_module(256, 256)
        self.pool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # (N, 256, 38, 38)
        self.conv_4_1 = VGG16Base.conv_module(256, 512)
        self.conv_4_2 = VGG16Base.conv_module(512, 512)
        self.conv_4_3 = VGG16Base.conv_module(512, 512) # Used as conv_4_3 features
        self.pool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 512, 19, 19)
        self.conv_5_1 = VGG16Base.conv_module(512, 512)
        self.conv_5_2 = VGG16Base.conv_module(512, 512)
        self.conv_5_3 = VGG16Base.conv_module(512, 512)
        self.pool_5_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # (N, 512, 19, 19)
        self.conv_6_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.act_6_1 = nn.ReLU(inplace=True)
        # (N, 1024, 19, 19)
        self.conv_7_1 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.act_7_1 = nn.ReLU(inplace=True) # Used as act_7_1 features

    @staticmethod
    def conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def decimate(self, tensor, m):
        assert tensor.dim() == len(m)
        for d in range(tensor.dim()):
            if m[d] is not None:
                indices = torch.arange(start=0, end=tensor.size(d), step=m[d]).long()
                tensor = tensor.index_select(dim=d, index=indices)
        return tensor

    def load_pretrained_weights(self):

        state_dict = self.state_dict()
        params = list(state_dict.keys())

        pretrained_state_dict = vgg16_bn(pretrained=True).state_dict()
        pretrained_params = list(pretrained_state_dict.keys())

        for i, param in enumerate(params[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_params[i]]

        # Convert FC6 weights to convolutional layers
        fc6_weight = pretrained_state_dict["classifier.0.weight"].view(4096, 512, 7, 7)
        fc6_bias = pretrained_state_dict["classifier.0.bias"]
        state_dict["conv_6_1.weight"] = self.decimate(fc6_weight, m=[4, None, 3, 3])
        state_dict["conv_6_1.bias"] = self.decimate(fc6_bias, m=[4])

        # Convert FC7 weights to convolutional layers
        fc7_weight = pretrained_state_dict["classifier.3.weight"].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict["classifier.3.bias"]
        state_dict["conv_7_1.weight"] = self.decimate(fc7_weight, m=[4, 4, None, None])
        state_dict["conv_7_1.bias"] = self.decimate(fc7_bias, m=[4])

        self.load_state_dict(state_dict)

#################################################################################
# Extra layers
#################################################################################

def conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True))

vgg16_extra = nn.Sequential(OrderedDict([
    ("extra_1_1", conv_module(1024, 256, kernel_size=1, padding=0)),
    ("extra_1_2", conv_module(256, 512, stride=2)),
    ("extra_2_1", conv_module(512, 128, kernel_size=1, padding=0)),
    ("extra_2_2", conv_module(128, 256, stride=2)),
    ("extra_3_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_3_2", conv_module(128, 256, padding=0)),
    ("extra_4_1", conv_module(256, 128, kernel_size=1, padding=0)),
    ("extra_4_2", conv_module(128, 256, padding=0)),
]))

#################################################################################
# Config
#################################################################################

vgg16_config = {
    "base": VGG16Base,
    "extra": vgg16_extra,
    "norms": nn.ModuleList([L2Norm(512)]),
    "n_channels": [512, 1024, 512, 256, 256, 256],
    "feature_layers": ["conv_4_3", "act_7_1", "extra_1_2", "extra_2_2", "extra_3_2", "extra_4_2"],
    "norm_layers": ["conv_4_3"]
}
