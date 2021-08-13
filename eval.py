import argparse
import cv2
import numpy as np
import os
import torch
from config import config
from core.datasets import readers
from core.models import models


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()

    # Load model
    output_dir = os.path.join("models", args.pid)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")

    # Use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read dataset
    Reader = readers[config.dataset.reader]
    dataset = Reader(config.dataset.train_path, target_transform=None)

    # Load model
    checkpoint = torch.load(os.path.join(checkpoints_dir, f"ckpt-{args.epoch}.tar"), map_location=device)
    model = models[config.model.type][config.model.name](n_classes=config.dataset.n_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    COLORS = np.random.uniform(0, 255, size=(len(dataset.classes), 3))
    for sample in dataset:
        with torch.no_grad():
            image = sample["image"]
            ground_truth = image.copy()
            targets = sample["targets"]

            input = model.preprocess(image)
            input = input.to(device)
            predictions = model(input)
            bboxs = model.postprocess(image, predictions)

            for x1, y1, x2, y2, conf, label in bboxs:
                x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2, label])
                cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[label - 1], 2)

            for x1, y1, x2, y2, label in targets:
                x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2, label])
                cv2.rectangle(ground_truth, (x1, y1), (x2, y2), COLORS[label - 1], 2)

            cv2.imshow("predictions", image)
            cv2.imshow("ground truth", ground_truth)
            if cv2.waitKey(0) == ord("q"):
                break


if __name__ == '__main__':
    main()
