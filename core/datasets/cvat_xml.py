import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

class CVATXMLTargetTransform(object):

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

class CVATXMLDataset(Dataset):

    def __init__(self, xml_path, transform=None, target_transform=CVATXMLTargetTransform()):
        self.root_dir = os.path.dirname(xml_path)
        self.xml_path = xml_path
        self.transform = transform
        self.target_transform = target_transform
        self.root = ET.parse(xml_path).getroot()
        self.classes = [x.find("name").text for x in self.root.iter("label")]
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

        self.image_paths = []
        self.annotations = []
        for data in self.root.iter("image"):

            image_path = os.path.join(self.root_dir, "images", "{}.jpg".format(data.attrib["id"]))
            self.image_paths.append(image_path)

            bboxs = []
            for bbox in data.iter("box"):
                bbox_info = bbox.attrib
                x1 = bbox_info["xtl"]
                y1 = bbox_info["ytl"]
                x2 = bbox_info["xbr"]
                y2 = bbox_info["ybr"]
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                label = self.classes.index(bbox_info["label"]) + 1
                bboxs.append([x1, y1, x2, y2, label])
            self.annotations.append(bboxs)
