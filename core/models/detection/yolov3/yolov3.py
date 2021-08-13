import cv2
import numpy as np
import torch
from core.utils.detection import nms
from core.models.classification.darknet.darknet53 import ConvLayerParam, YoloConvBlock, Darknet53


class YoloV3(torch.nn.Module):

    # Model Config
    config = {
        "name": "yolov3",
        "image_size": 416,
        "objectness_threshold": 0.5,
        "grid_sizes": [13, 26, 52],
        "num_anchors_per_grid_cell": 3,
        "anchors": [[116, 90,  156, 98,  373, 326],  [30, 61, 62, 45,  59, 119], [10, 13,  16, 30,  33, 23]],
        "nms_threshold": 0.5
    }

    def __init__(self, n_classes):
        super(YoloV3, self).__init__()
        self.n_classes = n_classes
        num_output_filters = (4 + 1 + n_classes) * self.config["num_anchors_per_grid_cell"]
        self.yolo_82 = None
        self.yolo_94 = None
        self.yolo_106 = None
        self.anchors = self._create_priors()

        self.backbone = Darknet53(n_classes)

        self.layers_75_to_79 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=1024, out_channels=512, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=1024, out_channels=512, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=1024, out_channels=512, kernel=1, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=False
        )
        self.layers_80_to_81 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=512, out_channels=1024, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=1024, out_channels=num_output_filters, kernel=1, stride=1, bn=False,
                               use_activation=False),
            ], use_skip_connection=False
        )
        self.layer_84 = YoloConvBlock(
            [ConvLayerParam(in_channels=512, out_channels=256, kernel=1, stride=1, bn=True, use_activation=True)],
            use_skip_connection=False
        )
        self.layer_85 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # layer86: concat operation
        self.layers_87_to_91 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=256+512, out_channels=256, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=256, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=256, kernel=1, stride=1, bn=True, use_activation=True)
            ], use_skip_connection=False
        )
        self.layers_92_to_93 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=256, out_channels=512, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=512, out_channels=num_output_filters, kernel=1, stride=1, bn=False,
                               use_activation=False),
            ], use_skip_connection=False
        )

        self.layer_96 = YoloConvBlock(
            [ConvLayerParam(in_channels=256, out_channels=128, kernel=1, stride=1, bn=True, use_activation=True)],
            use_skip_connection=False
        )
        self.layer_97 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # layer98: concat operation
        self.layers_99_to_105 = YoloConvBlock(
            [
                ConvLayerParam(in_channels=128+256, out_channels=128, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=128, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=128, kernel=1, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=128, out_channels=256, kernel=3, stride=1, bn=True, use_activation=True),
                ConvLayerParam(in_channels=256, out_channels=num_output_filters, kernel=1, stride=1, bn=False,
                               use_activation=False)
            ], use_skip_connection=False
        )

    def _create_priors(self):
        anchor_sizes = self.config["anchors"]
        grid_sizes = self.config["grid_sizes"]
        num_anchors_per_grid_cell = self.config["num_anchors_per_grid_cell"]
        image_size = self.config["image_size"]

        assert len(anchor_sizes) == len(grid_sizes)
        for anchor_size_list in anchor_sizes:
            assert len(anchor_size_list) == num_anchors_per_grid_cell * 2

        result_priors = []

        for s in range(len(anchor_sizes)):
            grid_size = grid_sizes[s]
            anchors = anchor_sizes[s]
            anchors = torch.tensor(anchors)
            anchors = anchors / image_size * grid_size  # re-scale anchors
            priors = torch.zeros(grid_size, grid_size, num_anchors_per_grid_cell, 4)

            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(num_anchors_per_grid_cell):
                        priors[i, j, k, 0] = j
                        priors[i, j, k, 1] = i
                        priors[i, j, k, 2] = anchors[k * 2 + 0]
                        priors[i, j, k, 3] = anchors[k * 2 + 1]

            result_priors.append(priors)

        return result_priors

    def forward(self, x):
        x = self.backbone(x, False)
        x = self.layers_75_to_79(x)
        self.yolo_82 = self.layers_80_to_81(x)
        x = self.layer_84(x)
        x = self.layer_85(x)
        x = torch.cat((x, self.backbone.skip_61), 1)     # layer 86
        x = self.layers_87_to_91(x)
        self.yolo_94 = self.layers_92_to_93(x)
        x = self.layer_96(x)
        x = self.layer_97(x)
        x = torch.cat((x, self.backbone.skip_36), 1)     # layer 98
        self.yolo_106 = self.layers_99_to_105(x)

        self.yolo_82 = self.yolo_82.permute(0, 2, 3, 1).contiguous()
        self.yolo_94 = self.yolo_94.permute(0, 2, 3, 1).contiguous()
        self.yolo_106 = self.yolo_106.permute(0, 2, 3, 1).contiguous()

        return [self.yolo_82, self.yolo_94, self.yolo_106]

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

        for yolo_output_index in range(len(predictions)):
            yolo_output = predictions[yolo_output_index]
            grid_size = self.config["grid_sizes"][yolo_output_index]
            image_size = self.config["image_size"]
            min_objectness_score = self.config["objectness_threshold"]
            nms_threshold = self.config["nms_threshold"]
            num_anchors_per_grid_cell = self.config["num_anchors_per_grid_cell"]
            batch_size = yolo_output.shape[0]
            # assert batch_size == 1

            # We will deal with tensors having shape
            # . . . (batch size, grid_size, grid_size, num_anchors_per_grid_cell, <data per box>)

            anchor_boxes_cxy = torch.unsqueeze(self.anchors[yolo_output_index], dim=0).\
                expand(batch_size, grid_size, grid_size, num_anchors_per_grid_cell, 4)

            yolo_output = yolo_output.view([batch_size, grid_size, grid_size, num_anchors_per_grid_cell, -1])

            # anchor_boxes_cxy = [anchor_x, anchor_y, anchor_height, anchor_width]
            # yolo_output = [x , y, w, h, objectness_score, <class scores...>]
            anchor_cx = anchor_boxes_cxy[:, :, :, :, 0]
            anchor_cy = anchor_boxes_cxy[:, :, :, :, 1]
            anchor_width = anchor_boxes_cxy[:, :, :, :, 2]
            anchor_height = anchor_boxes_cxy[:, :, :, :, 3]
            x = torch.sigmoid(yolo_output[:, :, :, :, 0])
            y = torch.sigmoid(yolo_output[:, :, :, :, 1])
            w = torch.tanh(yolo_output[:, :, :, :, 2])
            h = torch.tanh(yolo_output[:, :, :, :, 3])
            # w = yolo_output[:, :, :, :, 2]
            # h = yolo_output[:, :, :, :, 3]

            x = (anchor_cx + x) / grid_size
            y = (anchor_cy + y) / grid_size
            w = anchor_width * torch.exp(w) / grid_size
            h = anchor_height * torch.exp(h) / grid_size

            yolo_output[:, :, :, :, 4:] = torch.sigmoid(yolo_output[:, :, :, :, 4:])

            pred_conf = yolo_output[:, :, :, :, 4]
            confidence_scores = torch.unsqueeze(pred_conf, -1) * yolo_output[:, :, :, :, 5:]
            pred_classes_argmax = torch.argmax(confidence_scores, dim=-1)

            object_present = pred_conf > min_objectness_score
            result = []

            for image_index in range(batch_size):

                image_bboxs = []
                image_confs = []
                image_labels = []

                x_i = x[image_index]
                y_i = y[image_index]
                w_i = w[image_index]
                h_i = h[image_index]
                object_present_i = object_present[image_index]
                pred_classes_argmax_i = pred_classes_argmax[image_index]
                confidence_scores_i = confidence_scores[image_index]

                x_i = torch.unsqueeze(x_i[object_present_i], -1)
                y_i = torch.unsqueeze(y_i[object_present_i], -1)
                w_i = torch.unsqueeze(w_i[object_present_i], -1)
                h_i = torch.unsqueeze(h_i[object_present_i], -1)
                pred_classes_argmax_i = torch.unsqueeze(pred_classes_argmax_i[object_present_i], -1)
                confidence_scores_i = confidence_scores_i[object_present_i]

                if x_i.shape[0] == 0:
                    result.append(None)
                    continue

                bboxes = torch.cat((pred_classes_argmax_i, x_i - w_i / 2, y_i - h_i / 2, x_i + w_i / 2, y_i + h_i / 2,
                                    confidence_scores_i), dim=-1)

                for class_index in range(self.n_classes):
                    bbox_class_index = bboxes[bboxes[:, 0] == class_index]
                    if bbox_class_index.size(0) == 0:
                        continue

                    class_bboxs = bbox_class_index[:, 1: 5]
                    class_confs = bbox_class_index[:, 5 + class_index]

                    # Non-maximum suppression
                    class_confs, sort_idx = class_confs.sort(dim=0, descending=True)
                    class_bboxs = class_bboxs[sort_idx]
                    class_bboxs, class_confs = nms(class_bboxs, class_confs, max_overlap=nms_threshold)

                    image_bboxs.append(class_bboxs)
                    image_confs.append(class_confs)
                    image_labels.append(torch.Tensor(class_bboxs.size(0) * [class_index + 1]))
                    # . . . +1 because the source data has indices 1 based

                image_bboxs = torch.cat(image_bboxs, 0)
                image_confs = torch.cat(image_confs, 0)
                image_labels = torch.cat(image_labels, 0)

                # Keep only top K predictions
                if image_bboxs.size(0) > 100:
                    images_confs, sort_idx = image_confs.sort(dim=0, descending=True)
                    image_bboxs = image_bboxs[sort_idx][:100]
                    image_confs = image_confs[:100]
                    image_labels = image_labels[sort_idx][:100]

                # Scale bounding boxes
                height, width = image.shape[:2]
                image_bboxs *= torch.Tensor([width, height, width, height])

                result.append(torch.cat([image_bboxs, image_confs.unsqueeze(1), image_labels.unsqueeze(1)], 1))

            return result
