import math
import torch
import torch.nn.functional as F


def get_conv_module(in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool,
                    activation='', activation_params=None, bn=True):
    conv_layer = Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
    torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out')
    # conv_layer.weight *= 2.0    # scaling factor
    if conv_layer.bias is not None:
        torch.nn.init.constant_(conv_layer.bias, 0)

    modules = [conv_layer]
    if bn:
        modules.append(torch.nn.BatchNorm2d(out_channels))
    if activation:
        if activation_params:
            modules.append(getattr(torch.nn, activation)(*activation_params))
        else:
            modules.append(getattr(torch.nn, activation)())
    return torch.nn.Sequential(*modules)


def get_dw_conv_module(in_channels: int, kernel_size: int, stride: int, bias: bool,
                       activation='', activation_params=None):
    conv_layer = Conv2dDynamicSamePadding(in_channels, in_channels, kernel_size, stride=stride,
                                          bias=bias, groups=in_channels)
    torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out')
    # conv_layer.weight *= 2.0    # scaling factor
    if conv_layer.bias is not None:
        torch.nn.init.constant_(conv_layer.bias, 0)

    modules = [conv_layer, torch.nn.BatchNorm2d(in_channels)]
    if activation:
        if activation_params:
            modules.append(getattr(torch.nn, activation)(*activation_params))
        else:
            modules.append(getattr(torch.nn, activation)())
    return torch.nn.Sequential(*modules)


def rescale_num_filters(num_filters, width_scale_factor, depth_divisor):
    num_filters *= width_scale_factor
    new_filters = int(num_filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * num_filters:
        new_filters += depth_divisor
    return int(new_filters)


def rescale_num_repeats(num_repeats, depth_scale_factor):
    return int(math.ceil(depth_scale_factor * num_repeats))


# credit: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
# License: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE
def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


# Credit: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
# License: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/LICENSE
class Conv2dDynamicSamePadding(torch.nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
