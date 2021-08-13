import cv2
import numpy as np
import torch
from config import config
from torch import nn
from torchvision.models import resnet18

class Resnet18(nn.Module):

    # Model Config
    config = {
        "name": "resnet18",
        "image_size": 224
    }

    def __init__(self, n_classes):
        super(Resnet18, self).__init__()
        # Number of classes
        self.n_classes = n_classes
        self.base = resnet18(pretrained=True)
        # Freeze layers
        for param in self.base.parameters():
            param.requires_grad = False
        # Modify head to match n_classes
        num_features = self.base.fc.in_features
        self.base.fc = nn.Linear(num_features, self.n_classes)

    ####################################################################################################
    # Training methods
    ####################################################################################################
    def forward(self, x):
        return self.base(x)

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
        pass
