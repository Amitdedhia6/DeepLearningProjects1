import copy
import torch
import torch.nn.functional as F
from typing import List

from .MBConv import MBConvArgs, MBConvBlock
from .utils import get_conv_module, rescale_num_filters, rescale_num_repeats


blocks = [
    MBConvArgs(in_channels=32, out_channels=16, kernel_size=3, stride=1, in_channel_expand_ratio=1),
    MBConvArgs(in_channels=16, out_channels=24, kernel_size=3, stride=2, in_channel_expand_ratio=6),
    MBConvArgs(in_channels=24, out_channels=40, kernel_size=5, stride=2, in_channel_expand_ratio=6),
    MBConvArgs(in_channels=40, out_channels=80, kernel_size=3, stride=2, in_channel_expand_ratio=6),
    MBConvArgs(in_channels=80, out_channels=112, kernel_size=5, stride=1, in_channel_expand_ratio=6),
    MBConvArgs(in_channels=112, out_channels=192, kernel_size=5, stride=2, in_channel_expand_ratio=6),
    MBConvArgs(in_channels=192, out_channels=320, kernel_size=3, stride=1, in_channel_expand_ratio=6)
]

block_repeats = [1, 2, 2, 3, 3, 4, 1]


class EfficientNet(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        width_scale_factor = 1.0
        depth_scale_factor = 1.0
        default_resolution = 224
        drop_rate = None
        drop_connect_rate = None

        if self.config is not None:
            if "width_scale_factor" in self.config:
                width_scale_factor = self.config["width_scale_factor"]
            if "depth_scale_factor" in self.config:
                depth_scale_factor = self.config["depth_scale_factor"]
            if "image_size" in self.config:
                default_resolution = self.config["image_size"]
            if "drop_rate" in self.config:
                drop_rate = self.config["drop_rate"]
            if "drop_connect_rate" in self.config:
                drop_connect_rate = self.config["drop_connect_rate"]

        self.n_classes = n_classes
        self.width_scale_factor = width_scale_factor
        self.depth_scale_factor = depth_scale_factor
        self.input_image_size = default_resolution
        self.depth_divisor = 8
        self.drop_rate = drop_rate
        self.drop_connect_rate = drop_connect_rate

        num_filter_stem = rescale_num_filters(blocks[0].in_channels, self.width_scale_factor, self.depth_divisor)
        self.conv_stem = get_conv_module(in_channels=3, out_channels=num_filter_stem, kernel_size=3, stride=2,
                                         bias=False, activation='SiLU', activation_params=[True], bn=True)
        self.mb_conv_block_list: List[MBConvBlock] = []
        self._fill_mb_conv_block_list()
        self.mb_conv_block_list = torch.nn.ModuleList(self.mb_conv_block_list)

        top_in_filters = self.mb_conv_block_list[-1].block_args.out_channels
        top_out_filters = rescale_num_filters(1280, self.width_scale_factor, self.depth_divisor)
        self.conv_top = get_conv_module(in_channels=top_in_filters, out_channels=top_out_filters, kernel_size=1,
                                        stride=1, bias=False, activation='SiLU', activation_params=[True], bn=True)
        if self.drop_rate:
            self.drop_layer = torch.nn.Dropout(p=self.drop_rate)
        else:
            self.drop_layer = None
        self.dense_top = torch.nn.Linear(top_out_filters, self.n_classes)
        torch.nn.init.uniform_(self.dense_top.weight)
        # self.dense_top.weight.weight *= 1. / 3.    # scaling factor

    def _fill_mb_conv_block_list(self):
        total_blocks = sum(block_repeats)
        current_block = 0

        for i in range(len(block_repeats)):
            num_repeat = rescale_num_repeats(block_repeats[i], self.depth_scale_factor)
            for j in range(num_repeat):
                block_args = copy.deepcopy(blocks[i])
                in_filters = rescale_num_filters(block_args.in_channels, self.width_scale_factor, self.depth_divisor)
                out_filters = rescale_num_filters(block_args.out_channels, self.width_scale_factor, self.depth_divisor)
                if self.drop_connect_rate:
                    drop_connect_rate_calc = self.drop_connect_rate * float(current_block) / total_blocks
                    if drop_connect_rate_calc > 0.0001:     # i.e. if it is > 0
                        block_args.drop_connect_rate = drop_connect_rate_calc
                if j == 0:
                    block_args.in_channels = in_filters
                    block_args.out_channels = out_filters
                    self.mb_conv_block_list.append(MBConvBlock(block_args))
                else:
                    block_args.in_channels = out_filters
                    block_args.out_channels = out_filters
                    block_args.stride = 1
                    self.mb_conv_block_list.append(MBConvBlock(block_args))
                current_block += 1

    def forward(self, x):
        x = self.conv_stem(x)
        for mb_block in self.mb_conv_block_list:
            x = mb_block(x)
        x = self.conv_top(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), x.size(1))
        if self.drop_layer:
            x = self.drop_layer(x)
        x = self.dense_top(x)
        return x
