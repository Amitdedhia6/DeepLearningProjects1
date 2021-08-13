from typing import NamedTuple, List
import torch
from core.utils.common import Conv2dDynamicSamePadding


class ConvLayerParam(NamedTuple):
    in_channels: int
    out_channels: int
    kernel: int
    stride: int
    use_activation: bool
    bn: bool


class YoloConvLayer(torch.nn.Module):
    def __init__(self, param: ConvLayerParam):
        super().__init__()
        use_bias = False if param.bn else True
        if param.stride > 1:
            zero_pad = torch.nn.ZeroPad2d((1, 0, 1, 0))
            conv_layer = torch.nn.Conv2d(param.in_channels, param.out_channels, param.kernel,
                                         stride=param.stride, bias=use_bias)
            modules = [zero_pad, conv_layer]
        else:
            conv_layer = Conv2dDynamicSamePadding(param.in_channels, param.out_channels, param.kernel,
                                                  stride=param.stride, bias=use_bias)
            modules = [conv_layer]
        torch.nn.init.kaiming_normal_(conv_layer.weight, mode='fan_out')
        if param.bn:
            modules.append(torch.nn.BatchNorm2d(param.out_channels))
        if param.use_activation:
            modules.append(torch.nn.LeakyReLU())
        self.sequential = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)


class YoloConvBlock(torch.nn.Module):
    def __init__(self, params: List[ConvLayerParam], use_skip_connection: bool = True):
        super().__init__()
        self.skip_connection = use_skip_connection
        self.module_list = torch.nn.ModuleList()
        for p in params:
            self.module_list.append(YoloConvLayer(p))

    def forward(self, x):
        skip_layer_index = len(self.module_list) - 2
        current_layer_index = 0
        skip_layer = None
        for conv in self.module_list:
            if skip_layer_index == current_layer_index:
                skip_layer = x
            x = conv(x)
            current_layer_index += 1

        return skip_layer + x if self.skip_connection else x


class Darknet53(torch.nn.Module):
    # Model Config
    config = {
        "name": "darknet53",
        "image_size": 416
    }

    def __init__(self, n_classes):
        super(Darknet53, self).__init__()
        self.n_classes = n_classes
        self.skip_36 = None
        self.skip_61 = None

        self.layers_0_to_4 = YoloConvBlock(
            # 4 YoloConvLayer + 1 Add layer (due to skip connection) = total 5 layers
            [
                ConvLayerParam(in_channels=3, out_channels=32, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=32, out_channels=64, kernel=3, stride=2, bn=True, use_activation=True),
                ConvLayerParam(in_channels=64, out_channels=32, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=32, out_channels=64, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_5_to_8 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=64, out_channels=128, kernel=3, stride=2, bn=True, use_activation=True),
                ConvLayerParam(in_channels=128, out_channels=64, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=64, out_channels=128, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_9_to_11 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=128, out_channels=64, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=64, out_channels=128, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_12_to_15 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=2, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=128, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_16_to_36 = torch.nn.ModuleList()
        for i in range(7):
            self.layers_16_to_36.append(
                YoloConvBlock(
                    [
                        ConvLayerParam(in_channels=256, out_channels=128, kernel=1, stride=1, bn=True,
                                       use_activation=True),
                        ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=1, bn=True,
                                       use_activation=True)
                    ], use_skip_connection=True
                )
            )
        self.layers_37_to_40 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=2, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=256, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_41_to_61 = torch.nn.ModuleList()
        for i in range(7):
            self.layers_41_to_61.append(
                YoloConvBlock(
                    [
                        ConvLayerParam(in_channels=512, out_channels=256, kernel=1, stride=1, bn=True,
                                       use_activation=True),
                        ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=1, bn=True,
                                       use_activation=True)
                    ], use_skip_connection=True
                )
            )
        self.layers_62_to_65 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=2, bn=True, use_activation=True),
                ConvLayerParam(in_channels=1024, out_channels=512, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=True
        )
        self.layers_66_to_74 = torch.nn.ModuleList()
        for i in range(3):
            self.layers_66_to_74.append(
                YoloConvBlock(
                    [
                        ConvLayerParam(in_channels=1024, out_channels=512, kernel=1, stride=1, bn=True,
                                       use_activation=True),
                        ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=1, bn=True,
                                       use_activation=True)
                    ], use_skip_connection=True
                )
            )
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(1024, self.n_classes)

    def forward(self, x, classification_mode=True):
        x = self.layers_0_to_4(x)
        x = self.layers_5_to_8(x)
        x = self.layers_9_to_11(x)
        x = self.layers_12_to_15(x)
        for layer in self.layers_16_to_36:
            x = layer(x)
        self.skip_36 = x
        x = self.layers_37_to_40(x)
        for layer in self.layers_41_to_61:
            x = layer(x)
        self.skip_61 = x
        x = self.layers_62_to_65(x)
        for layer in self.layers_66_to_74:
            x = layer(x)

        if classification_mode:
            x = self.global_avg_pool(x)
            x = x.view(-1, 1024)
            x = self.fc(x)

        return x

