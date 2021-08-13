import cv2
import itertools
import numpy as np
import torch
from .base.mobilenetv2 import mobilenetv2_config
from ....utils.detection import gcxgcy_to_xyxy
from ....utils.detection import nms
from config import config
from math import sqrt
from torch import nn
from torch.nn import functional as F


class DepthwiseSeparableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConvLayer, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = nn.BatchNorm2d(out.shape[1])(out)
        nn.ReLU(inplace=True)(out)
        out = self.pointwise(out)
        return out


def conv_module(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return DepthwiseSeparableConvLayer(in_channels, out_channels, kernel_size, stride, padding)


class SSDLite300(nn.Module):

    # Model Config
    config = {
        "name": "SSDLite300",
        "image_size": 300,
        "dims": [19, 10, 5, 3, 2, 1],
        "aspect_ratios": [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        "scales": [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05],
        "variances": [0.1, 0.2]
    }

    config.update(mobilenetv2_config)

    def __init__(self, n_classes):
        super(SSDLite300, self).__init__()
        # Number of classes (Including background as 0)
        self.n_classes = n_classes

        self.base = self.config["base"]()
        self.extra = self.config["extra"]
        self.norms = self.config.get("norms", None)
        self.feature_layers = self.config["feature_layers"]
        self.norm_layers = self.config.get("norm_layers", [])

        self.locs = nn.ModuleList([])
        self.confs = nn.ModuleList([])
        n_channels = self.config["n_channels"]
        for i, ar in enumerate(self.config["aspect_ratios"]):
            n = len(ar) * 2 + 2
            self.locs.append(conv_module(n_channels[i], n * 4, kernel_size=3, padding=1))
            self.confs.append(conv_module(n_channels[i], n * self.n_classes, kernel_size=3, padding=1))

        # Model Priors
        self.priors = self._create_prior_boxes()

        # Initialization
        self._initialize_weights(self.children())
        self.base.load_pretrained_weights()

    ####################################################################################################
    # Private methods
    ####################################################################################################
    def _initialize_weights(self, layers):
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            else:
                try:
                    iterator = layer.children()
                except TypeError:
                    # not iterable
                    pass
                else:
                    self._initialize_weights(iterator)

    def _create_prior_boxes(self):
        prior_boxes = []
        for k, dim in enumerate(self.config["dims"]):
            for i, j in itertools.product(range(dim), repeat=2):
                # prior box center
                cx = (j + 0.5) / dim
                cy = (i + 0.5) / dim
                # prior scale
                scale = self.config["scales"][k]
                additional_scale = sqrt(scale * self.config["scales"][k + 1])
                # aspect ratio: 1
                prior_boxes.append([cx, cy, scale, scale])
                prior_boxes.append([cx, cy, additional_scale, additional_scale])
                # Rest of the aspect ratios
                for ratio in self.config["aspect_ratios"][k]:
                    prior_boxes.append([cx, cy, scale * sqrt(ratio), scale / sqrt(ratio)])
                    prior_boxes.append([cx, cy, scale / sqrt(ratio), scale * sqrt(ratio)])
        prior_boxes = torch.FloatTensor(prior_boxes).view(-1, 4)
        return prior_boxes

    ####################################################################################################
    # Training methods
    ####################################################################################################
    def forward(self, x):
        # Generate low and high level features
        features = []
        for name, m in itertools.chain(self.base._modules.items(), self.extra._modules.items()):
            # Forward module
            x = m(x)
            # Save feature maps from different layers
            if name in self.feature_layers:
                if name in self.norm_layers:
                    idx = self.norm_layers.index(name)
                    features.append(self.norms[idx](x))
                else:
                    features.append(x)
        # Generate predictions from features
        locs, confs = [], []
        for i, feature in enumerate(features):
            loc = self.locs[i](feature)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(loc.size(0), -1, 4)
            locs.append(loc)

            conf = self.confs[i](feature)
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(conf.size(0), -1, self.n_classes)
            confs.append(conf)

        locs = torch.cat(locs, dim=1)
        confs = torch.cat(confs, dim=1)
        return locs, confs


    ####################################################################################################
    # Inference methods
    ####################################################################################################
    def preprocess(self, image):
        image = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        return image

    def postprocess(self, image, predictions):
        predictions = [pred.detach().cpu() for pred in predictions]
        predict_locs, predict_confs = predictions

        assert predict_locs.size(0) == 1
        predict_locs = predict_locs[0]
        predict_confs = predict_confs[0]

        predict_confs = F.softmax(predict_confs, dim=1)
        predict_bboxs = gcxgcy_to_xyxy(predict_locs, self.priors, variances=self.config["variances"])

        image_bboxs = []
        image_confs = []
        image_labels = []

        for cls in range(1, self.n_classes):
            # Class confidences
            class_confs = predict_confs[:, cls]

            # Get bboxs that exceeds threshold
            exceed_conf_threshold = class_confs > 0.5
            class_confs = class_confs[exceed_conf_threshold]
            class_bboxs = predict_bboxs[exceed_conf_threshold]

            if class_confs.size(0) == 0:
                continue

            # Non-maximum suppression
            class_confs, sort_idx = class_confs.sort(dim=0, descending=True)
            class_bboxs = class_bboxs[sort_idx]
            class_bboxs, class_confs = nms(class_bboxs, class_confs, max_overlap=0.2)

            image_bboxs.append(class_bboxs)
            image_confs.append(class_confs)
            image_labels.append(torch.Tensor(class_bboxs.size(0) * [cls]))

        if len(image_bboxs) == 0: return []

        image_bboxs = torch.cat(image_bboxs, 0)
        image_confs = torch.cat(image_confs, 0)
        image_labels = torch.cat(image_labels, 0)

        # Keep only top K predictions
        if image_bboxs.size(0) > 100:
            images_confs, sort_idx = images_confs.sort(dim=0, descending=True)
            image_bboxs = image_bboxs[sort_idx][:100]
            image_confs = image_confs[:100]
            image_labels = image_labels[sort_idx][:100]

        # Scale bounding boxes
        height, width = image.shape[:2]
        image_bboxs *= torch.Tensor([width, height, width, height])

        return torch.cat([image_bboxs, image_confs.unsqueeze(1), image_labels.unsqueeze(1)], 1)
