import argparse
import cv2
import numpy as np
import os
import torch
from config import config
from core.augmentations import detection as detection_augmentations
from core.datasets import readers
from core.logging import configure_logger
from core.metrics import mean_average_precision
from core.models.detection import SSD300
from torch.utils.data import DataLoader
from torchvision import transforms

def main():

    logger = configure_logger()

    train_path = "coco2017/train.txt"
    n_classes = 6
    model_path = os.path.join("models", "ckpt-500.tar")

    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model from path ...")
    checkpoint = torch.load(model_path)
    model = SSD300(n_classes=n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Read dataset
    Reader = readers["yolov4"]

    ############################################################################
    # The following code is use to visualize model output and ground truth
    # Comment the this block when implementing mAP
    ############################################################################
    dataset = Reader(train_path, target_transform=None)
    colors = np.random.uniform(0, 255, size=(len(dataset.classes), 3))
    for sample in dataset:

        image = sample["image"]
        ground_truth = image.copy()
        targets = sample["targets"]

        input = model.preprocess(image)
        input = input.to(device)
        predictions = model(input)
        bboxs = model.postprocess(image, predictions)

        for x1, y1, x2, y2, conf, label in bboxs:
            x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2, label])
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[label - 1], 2)

        for x1, y1, x2, y2, label in targets:
            x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2, label])
            cv2.rectangle(ground_truth, (x1, y1), (x2, y2), colors[label - 1], 2)

        cv2.imshow("predictions", image)
        cv2.imshow("ground truth", ground_truth)
        if cv2.waitKey(0) == ord("q"):
            break


    ############################################################################
    # The following code is use to compute mAP
    ############################################################################
    # transform = transforms.Compose([
    #     detection_augmentations.Resize(width=model.config["image_size"], height=model.config["image_size"]),
    #     detection_augmentations.ToTensor()
    # ])
    # dataset = Reader(train_path, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=Reader.collate_fn)
    #
    # ious = np.arange(0.5, 1.0, 0.05)
    # map = 0
    # for iou in ious:
    #     logger.info(f"Computing AP for IOU {iou} ...")
    #     iou_map = 0
    #     with torch.no_grad():
    #         for i, (images, targets) in enumerate(dataloader):
    #             images = images.to(device)
    #             predictions = model(images)
    #             targets = [target.to(device) for target in targets]
    #             iou_map += mean_average_precision(predictions, targets, iou)
    #     iou_map /= len(dataloader)
    #     map += iou_map
    # map /= len(ious)



if __name__ == '__main__':
    main()
