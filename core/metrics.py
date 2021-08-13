import torch
import numpy as np
from .utils.detection import get_jaccard_overlaps
from collections import Counter
from pprint import pprint


def mean_average_precision_with_iou(targets, predictions, n_classes, device, iou_thresh=0.5):

    class_precisions = []
    for c in range(1, n_classes + 1):
        print(f"Processing class: {c}")

        # Get all targets of same class
        class_targets = targets[targets[:, 5] == c]
        n_class_targets = len(class_targets)

        print("class targets")
        print(class_targets)
        print()

        # Keep track of which target has already been 'detected'
        target_counter = Counter([int(target[0].item()) for target in class_targets])
        target_dict = { image_id: torch.zeros(n_bbox) for image_id, n_bbox in target_counter.items()}

        # Get all prediction of same class
        class_predictions = predictions[predictions[:, 6] == c]
        if len(class_predictions) == 0: continue

        # Sort class predictions by confidence
        class_predictions = class_predictions[class_predictions[:, 5].sort(descending=True)[1]]

        # Save TPs and FPs
        TP = torch.zeros((len(class_predictions)), dtype=torch.float).to(device)
        FP = torch.zeros((len(class_predictions)), dtype=torch.float).to(device)

        for i, class_prediction in enumerate(class_predictions):

            print("Class prediction")
            print(class_prediction)

            # Get all targets with same class from the same image
            image_id = int(class_prediction[0].item())
            image_class_targets = class_targets[class_targets[:, 0] == image_id]

            print("Image class target")
            print(image_class_targets)
            if len(image_class_targets) == 0:
                FP[i] = 1
                continue
            # Compute IOU
            overlaps = get_jaccard_overlaps(class_prediction[1:5].unsqueeze(0), image_class_targets[:, 1:5])
            print("overlaps")
            print(overlaps)

            max_overlap, max_overlap_index = torch.max(overlaps.squeeze(0), dim=0)
            print(f"Maximum overlap {max_overlap.item()} at index {max_overlap_index.item()}")

            pprint(target_dict)
            if max_overlap.item() > iou_thresh:
                if target_dict[image_id][max_overlap_index.item()] == 0:
                    print("Add TP")
                    target_dict[image_id][max_overlap_index.item()] = 1
                    TP[i] = 1
                else:
                    print("Add FP")
                    FP[i] = 1
            else:
                print("Add FP")
                FP[i] = 1

            print()


        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (n_class_targets + 1e-6)
        print(f"Recall: {recalls}")
        recalls = torch.cat((torch.tensor([0]).to(device), recalls))
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        print(f"Precision: {precisions}")
        precisions = torch.cat((torch.tensor([1]).to(device), precisions))
        class_precisions.append(torch.trapz(precisions, recalls))

    average_precision = sum(class_precisions) / len(class_precisions)
    print(f"AP @ IOU {iou_thresh:.2f}: {average_precision:.5f}")
    return average_precision


def mean_average_precision(targets, predictions, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store all targets in continuous tensor keeping image_id
    targets = [np.c_[np.ones(target.shape[0]) * i, target] for i, target in enumerate(targets)]
    targets = np.concatenate(targets, 0)
    targets = torch.from_numpy(targets).to(device)
    # Store all prediction in continuous tensor keeping image_id
    predictions = [np.c_[np.ones(pred.shape[0]) * i, pred] for i, pred in enumerate(predictions) if len(pred) > 0]
    predictions = np.concatenate(predictions, 0)
    predictions = torch.from_numpy(predictions).to(device)

    print("Targets")
    print(targets)

    print("Predicitions")
    print(predictions)

    iou_precisions = []
    for iou_thresh in [0.5]: # np.arange(0.5, 1.0, 0.05):
        precision = mean_average_precision_with_iou(targets, predictions, len(classes), device, iou_thresh=iou_thresh)
        iou_precisions.append(precision)

    mean_average_precision_val = sum(iou_precisions) / len(iou_precisions)
    print(f"Mean AP: {mean_average_precision_val:.5f}")
    return mean_average_precision_val.item()
