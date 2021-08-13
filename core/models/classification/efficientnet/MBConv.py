import torch
import torch.nn.functional as F

from .utils import get_dw_conv_module, get_conv_module, drop_connect


class MBConvArgs:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 in_channel_expand_ratio: int, se_ratio: float = 0.25, id_skip: bool = True,
                 drop_connect_rate: float = None):
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.in_channel_expand_ratio: int = in_channel_expand_ratio
        self.se_ratio: float = se_ratio
        self.id_skip: bool = id_skip
        self.drop_connect_rate = drop_connect_rate


class MBConvBlock(torch.nn.Module):
    def __init__(self, block_args: MBConvArgs):
        super().__init__()
        self.block_args: MBConvArgs = block_args

        # Expansion phase
        expanded_num_channels = block_args.in_channels * block_args.in_channel_expand_ratio
        if block_args.in_channel_expand_ratio != 1:
            self.conv_expand = get_conv_module(in_channels=self.block_args.in_channels,
                                               out_channels=expanded_num_channels,
                                               kernel_size=1, stride=1, bias=False,
                                               activation='SiLU', activation_params=[True], bn=True)
        else:
            self.conv_expand = None

        # Depthwise Convolution
        self.dw_conv = get_dw_conv_module(in_channels=expanded_num_channels,
                                          kernel_size=self.block_args.kernel_size,
                                          stride=self.block_args.stride, bias=False,
                                          activation='SiLU', activation_params=[True])

        # Squeeze and Excitation
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.in_channels * self.block_args.se_ratio))
            self.conv_squeeze = get_conv_module(in_channels=expanded_num_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1, stride=1, bias=True,
                                                activation='SiLU', activation_params=[True], bn=False)
            self.conv_excite = get_conv_module(in_channels=num_squeezed_channels,
                                               out_channels=expanded_num_channels,
                                               kernel_size=1, stride=1, bias=True,
                                               activation='Sigmoid', bn=False)

        self.conv_output = get_conv_module(in_channels=expanded_num_channels,
                                           out_channels=self.block_args.out_channels,
                                           kernel_size=1, stride=1, bias=False,
                                           activation=None, bn=True)

    def forward(self, x):
        if self.conv_expand:
            y = self.conv_expand(x)
        else:
            y = x
        y = self.dw_conv(y)
        if self.has_se:
            z = F.adaptive_avg_pool2d(y, 1)
            z = self.conv_squeeze(z)
            z = self.conv_excite(z)
            y = y * z

        y = self.conv_output(y)

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.in_channels, self.block_args.out_channels
        drop_connect_rate = self.block_args.drop_connect_rate
        if self.block_args.id_skip and self.block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate is not None:
                y = drop_connect(y, p=drop_connect_rate, training=self.training)
            y = y + x  # skip connection

        return y
