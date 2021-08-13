import torch

class EfficientnetB0(torch.nn.Module):

    # Model Config
    config = {
        "name": "efficientnetb0",
        "image_size": 224
    }

    def __init__(self, n_classes):
        super(EfficientnetB0, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        pass
