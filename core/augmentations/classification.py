from torchvision import transforms as T
from random import random


class RandomGaussianBlur(object):

    def __init__(self, kernel_size=3, p=0.5):
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=kernel_size)

    def __call__(self, image):
        if random() < self.p:
            return self.blur(image)
        else:
            return image


class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.p = p
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image):
        if random() < self.p:
            return self.jitter(image)
        else:
            return image


class RandomRotation(object):
    def __init__(self, degrees=(0, 180), p=0.5):
        self.p = p
        self.rotate = T.RandomRotation(degrees=degrees)

    def __call__(self, image):
        if random() < self.p:
            return self.rotate(image)
        else:
            return image