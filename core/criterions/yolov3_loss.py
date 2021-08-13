import torch
from torch import nn
from ..utils.detection import xyxy_to_cxcy, cxcy_to_xyxy, get_jaccard_overlaps


class YoloV3Loss(nn.Module):
    def __init__(self, model_config, anchors, device):
        super(YoloV3Loss, self).__init__()
        self.model_config = model_config
        self.device = device
        self.anchors = anchors      # priors in cxcy format
        for i in range(len(self.anchors)):
            self.anchors[i] = self.anchors[i].to(device)

        # Loss functions
        self.mse = nn.MSELoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")



    def forward(self, predictions, gt_bbox_list):
        # predictions: A list of 3 outputs of Yolo Architecture.
        # . . . Each list is a tensor of size:  BatchSize X GridSize X GridSize X
        # . . . . . .(4 + 1 + n_classes) * num_anchor_box_per_grid_cell
        # . . . E.g. First item may of size B X 13 X 13 X 30 (where n_classes = 5)
        # gt_bbox_list: A list of BatchSize items - one per image, each tensor of size n X 5 where n is number of
        # . . . objects in the image and 2nd dimension has format (xmin, ymin, xmax, ymax, class (1 based)), x and y
        # . . . are normalized between 0 and 1

        batch_size = predictions[0].shape[0]
        num_anchors_per_grid_cell = self.model_config["num_anchors_per_grid_cell"]

        loss_x = torch.zeros(1).to(self.device)
        loss_y = torch.zeros(1).to(self.device)
        loss_w = torch.zeros(1).to(self.device)
        loss_h = torch.zeros(1).to(self.device)
        loss_conf_obj = torch.zeros(1).to(self.device)
        loss_conf_noobj = torch.zeros(1).to(self.device)
        loss_class = torch.zeros(1).to(self.device)
        min_iou_for_objectness = self.model_config["objectness_threshold"]

        for yolo_output_index in range(len(predictions)):
            grid_size = self.model_config["grid_sizes"][yolo_output_index]
            yolo_output = predictions[yolo_output_index]
            yolo_output = yolo_output.view(batch_size, grid_size, grid_size, num_anchors_per_grid_cell, -1)

            # First declare all tensors required for loss calculation
            # . . .in below declarations its interpreted as (..., number of grid columns, number of grid rows, ...)
            object_mask = torch.zeros(batch_size, grid_size, grid_size, num_anchors_per_grid_cell,
                                      dtype=torch.bool).to(self.device)
            no_object_mask = torch.ones(batch_size, grid_size, grid_size, num_anchors_per_grid_cell,
                                        dtype=torch.bool).to(self.device)
            class_mask = torch.ones(batch_size, grid_size, grid_size, num_anchors_per_grid_cell,
                                    dtype=torch.int64).to(self.device)

            box_loss_scale = torch.ones(batch_size, grid_size, grid_size, num_anchors_per_grid_cell).to(self.device)

            tx = torch.zeros(batch_size, grid_size, grid_size, num_anchors_per_grid_cell).to(self.device)
            ty = torch.zeros(batch_size, grid_size, grid_size, num_anchors_per_grid_cell).to(self.device)
            tw = torch.zeros(batch_size, grid_size, grid_size, num_anchors_per_grid_cell).to(self.device)
            th = torch.zeros(batch_size, grid_size, grid_size, num_anchors_per_grid_cell).to(self.device)

            anchor_boxes_cxy = self.anchors[yolo_output_index]
            anchor_boxes_xy = (cxcy_to_xyxy(anchor_boxes_cxy.view(-1, 4))).view(grid_size, grid_size,
                                                                                num_anchors_per_grid_cell, 4)
            # . . . anchor_boxes dim = (grid_size, grid_size, num_anchors_per_grid_cell, 4)

            for image_index in range(len(gt_bbox_list)):
                gt_bbox_image_xy = gt_bbox_list[image_index].clone()            # tensor of GT bbox for one image: n X 5
                gt_bbox_image_xy[:, :4] = gt_bbox_image_xy[:, :4] * grid_size   # normalize to grid cells
                gt_bbox_image_xy[:, 4] = gt_bbox_image_xy[:, 4] - 1             # The class is 1 based, make it 0 based

                gt_bbox_image_cxy = gt_bbox_image_xy.clone()
                gt_bbox_image_cxy[:, 0:4] = xyxy_to_cxcy(gt_bbox_image_cxy[:, 0:4])

                gt_box_loss_scale = 2 - gt_bbox_image_cxy[:, 2] * gt_bbox_image_cxy[:, 3] / (grid_size * grid_size)
                # . . . divide by grid size because the gt_bbox_image_cxy has been normalized to grid size

                overlaps = get_jaccard_overlaps(gt_bbox_image_xy[:, :4], anchor_boxes_xy.view(-1, 4))
                overlaps_object_present = (overlaps >= min_iou_for_objectness).nonzero(as_tuple=False)
                for i in range(overlaps_object_present.shape[0]):
                    # x = overlaps_object_present[i, 0].item()    # x is GT bbox index
                    y = overlaps_object_present[i, 1].item()    # y is index into overlaps tensor declared before

                    grid_row_index = int((y // num_anchors_per_grid_cell) / grid_size)
                    grid_col_index = (y // num_anchors_per_grid_cell) % grid_size
                    anchor_index = y % num_anchors_per_grid_cell
                    no_object_mask[image_index, grid_row_index, grid_col_index, anchor_index] = 0

                for gt_bbox_index in range(gt_bbox_image_xy.shape[0]):
                    gt_bbox = gt_bbox_image_cxy[gt_bbox_index][0:4]
                    grid_row_index = int(torch.floor(gt_bbox[1]).item())    # the GT bbox center falls in grid cell
                    grid_col_index = int(torch.floor(gt_bbox[0]).item())

                    start_index = (grid_col_index + grid_row_index * grid_size) * num_anchors_per_grid_cell
                    overlaps_gt_grid_cell = overlaps[gt_bbox_index, start_index:start_index + num_anchors_per_grid_cell]
                    overlap_max_values, overlap_max_indices = overlaps_gt_grid_cell.max(dim=0)
                    overlap_val = overlap_max_values.item()
                    anchor_index = overlap_max_indices.item()
                    if overlap_val >= min_iou_for_objectness:
                        box_loss_scale[image_index, grid_row_index, grid_col_index, anchor_index] = \
                            gt_box_loss_scale[gt_bbox_index]
                        object_mask[image_index, grid_row_index, grid_col_index, anchor_index] = 1
                        class_mask[image_index, grid_row_index, grid_col_index, anchor_index] = \
                            int(gt_bbox_image_xy[gt_bbox_index][-1].item())
                        tx[image_index, grid_row_index, grid_col_index, anchor_index] = (gt_bbox[0] -
                                                                                         torch.floor(gt_bbox[0])).item()
                        ty[image_index, grid_row_index, grid_col_index, anchor_index] = (gt_bbox[1] -
                                                                                         torch.floor(gt_bbox[1])).item()
                        tw[image_index, grid_row_index, grid_col_index, anchor_index] = \
                            torch.log(gt_bbox[2] / anchor_boxes_cxy[grid_row_index, grid_col_index, anchor_index][2] + 1e-16)
                        th[image_index, grid_row_index, grid_col_index, anchor_index] = \
                            torch.log(gt_bbox[3] / anchor_boxes_cxy[grid_row_index, grid_col_index, anchor_index][3] + 1e-16)

            tconf = object_mask.float()
            x = torch.sigmoid(yolo_output[:, :, :, :, 0])
            y = torch.sigmoid(yolo_output[:, :, :, :, 1])
            w = torch.tanh(yolo_output[:, :, :, :, 2])
            h = torch.tanh(yolo_output[:, :, :, :, 3])
            # w = yolo_output[:, :, :, :, 2]
            # h = yolo_output[:, :, :, :, 3]
            pred_conf = torch.sigmoid(yolo_output[:, :, :, :, 4])
            pred_class = yolo_output[:, :, :, :, 5:]

            loss_x += sum(self.mse(x[object_mask], tx[object_mask]) * box_loss_scale[object_mask])
            loss_y += sum(self.mse(y[object_mask], ty[object_mask]) * box_loss_scale[object_mask])
            loss_w += sum(self.mse(w[object_mask], tw[object_mask]) * box_loss_scale[object_mask])
            loss_h += sum(self.mse(h[object_mask], th[object_mask]) * box_loss_scale[object_mask])
            loss_class += self.cross_entropy(pred_class[object_mask], class_mask[object_mask])
            loss_conf_obj += self.bce(pred_conf[object_mask], tconf[object_mask])
            loss_conf_noobj += self.bce(pred_conf[no_object_mask], tconf[no_object_mask])

        return loss_x + loss_y + loss_w + loss_h + loss_class + (loss_conf_obj + loss_conf_noobj)
