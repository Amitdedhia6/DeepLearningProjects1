import cv2
import numpy as np
import os
import torch
import json
from torch.utils.data import Dataset
from collections import defaultdict

class COCOJSONTargetTransform(object):

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        height, width = image.shape[:2]
        # Normalized x1, y1, x2, y2
        bboxs = targets[:, :4].copy()
        bboxs[:, 0::2] /= width
        bboxs[:, 1::2] /= height
        # Label
        labels = targets[:, -1].copy()
        labels = np.expand_dims(labels, 1)
        targets = np.hstack((bboxs, labels))
        return { "image": image, "targets": targets }

class COCOJSONDataset(Dataset):

    def __init__(self, json_path, transform=None, target_transform=COCOJSONTargetTransform()):
        self.root_dir = os.path.dirname(json_path)
        self.json_path = json_path
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = json.load(open(self.json_path, "r"))
        self.classes = [x["name"] for x in self.dataset["categories"]]
        self._create_index()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = cv2.imread(self.image_paths[idx])
        bboxs = self.annotations[idx]

        targets = np.zeros((0, 5))
        if len(bboxs) == 0:
            return targets

        for bbox in bboxs:
            target = np.zeros((1, 5))
            target[0, 0] = bbox[0] # x1
            target[0, 1] = bbox[1] # y1
            target[0, 2] = bbox[2] # x2
            target[0, 3] = bbox[3] # y2
            target[0, 4] = bbox[4] # label (0 is assign to background)
            targets = np.append(targets, target, axis=0)

        sample = { "image": image, "targets": targets }

        if self.target_transform is not None:
            sample = self.target_transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        targets_list = []
        images = []
        for sample in batch:
            image, targets = sample["image"], sample["targets"]
            images.append(image)
            targets_list.append(targets)
        return (torch.stack(images, 0), targets_list)

    def _create_index(self):

        self.image_paths = {}
        self.annotations = defaultdict(list)

        for image in self.dataset["images"]:

            image_path = os.path.join(self.root_dir, "images", image["file_name"])
            
            self.image_paths[image["id"]] = image_path

        for annotation in self.dataset["annotations"]:
            x, y, w, h = annotation["bbox"] 
            label = annotation["category_id"]
            bbox = [x, y, x+w, y+h, label]
            self.annotations[annotation["image_id"]].append(bbox)
