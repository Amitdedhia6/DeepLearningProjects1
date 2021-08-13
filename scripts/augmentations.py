import cv2
import numpy as np
import torch
from config import config
from core.augmentations import detection as detection_augmentations
from core.datasets import readers
from core.logging import configure_logger
from pprint import pformat
from torch.utils.data import DataLoader
from torchvision import transforms

def main():

    logger = configure_logger()

    reader = "cvat_xml"
    train_path = "dataset/train.xml"

    # Read datasets
    Reader = readers[reader]
    transform = transforms.Compose([
        detection_augmentations.Resize(width=300, height=300), # Comment this if using LetterboxResize
        # augmentations.detection.LetterboxResize(width=300, height=300),
        detection_augmentations.RandomFlip(),
        detection_augmentations.RandomBlur(),
        detection_augmentations.RandomCutout(),
        detection_augmentations.RandomHSV(0.5, 0.5, 0.5),
        detection_augmentations.ToTensor()
    ])

    logger.info(f"Reading training data from {train_path} (reader) ...")
    train_dataset = Reader(train_path, transform=transform)

    logger.info(f"Classes: {', '.join(train_dataset.classes)}", extra={ "type": "DATASET" })
    logger.info(f"Training samples: {len(train_dataset)}", extra={ "type": "DATASET" })
    colors = np.random.uniform(0, 255, size=(len(train_dataset.classes), 3))

    train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=Reader.collate_fn)

    for images, targets in train_dataloader:

        image = images.squeeze(0)
        image = image.permute(1, 2, 0).contiguous().numpy()
        image = np.uint8(image)

        image_h, image_w = image.shape[:2]
        for target in targets[0]:
            x1, y1, x2, y2, label = target
            x1 = int(image_w * x1)
            y1 = int(image_h * y1)
            x2 = int(image_w * x2)
            y2 = int(image_h * y2)
            label = int(label) - 1

            image = cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 1)

        cv2.imshow("image", image)
        if cv2.waitKey(0) == ord('q'): break

if __name__ == '__main__':
    main()
