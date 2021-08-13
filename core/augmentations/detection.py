import cv2
import itertools
import numpy as np
import torch
from random import random
from torchvision import transforms


class LetterboxResize(object):

    def __init__(self, width, height):
        self.image_dim = (width, height)

    def __call__(self, sample):
        # TODO (TK): Implement this  --> DONE
        image, targets = sample["image"], sample["targets"]
        ih, iw, num_channels = image.shape
        ew, eh = self.image_dim

        # Calculate the scale needed
        scale = min(eh / ih, ew / iw)
        nw, nh = int(scale * iw), int(scale * ih)

        image = cv2.resize(image, (nw, nh))
        new_img = np.full((eh, ew, num_channels), 0, dtype='uint8')

        # fill new image with the resized image and centered it
        new_img[(eh - nh) // 2 : (eh - nh) // 2 + nh,
                (ew - nw) // 2 : (ew - nw) // 2 + nw,
               :] = image.copy()

        # Coordinate transformation for bounding boxes
        targets[:, 0] = (targets[:, 0] * nw + (ew - nw) // 2) / ew
        targets[:, 1] = (targets[:, 1] * nh + (eh - nh) // 2) / eh
        targets[:, 2] = (targets[:, 2] * nw + (ew - nw) // 2) / ew
        targets[:, 3] = (targets[:, 3] * nh + (eh - nh) // 2) / eh

        return { "image": new_img, "targets": targets }

class RandomBlur(object):

    def __init__(self, blur_type="normal", kernel_size=(3,3), p=0.5):
        self.p = p
        self.kernel_size = kernel_size
        self.blur_type = blur_type

    def __call__(self, sample):
        # TODO (TK): Implement this  --> DONE
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]
            switcher = {
                "normal": cv2.blur(image, self.kernel_size),
                "gaussian": cv2.GaussianBlur(image, self.kernel_size, 0),
                "median": cv2.medianBlur(image, self.kernel_size[0]),
            }
            image = switcher[self.blur_type]
            return { "image": image, "targets": targets }
        else:
            return sample

class RandomCutout(object):

    def __init__(self, p=0.5):
        # Scales: [1/2] * 1 + [1/4] * 2 + [1/8] * 4 + [1/16] * 8 + [1/32] * 16
        self.scales = [[ 1 / (2 ** (i + 1))] * (2 ** i) for i in range(4)]
        self.scales = list(itertools.chain.from_iterable(self.scales))
        self.p = p

    def __call__(self, sample):
        # TODO (TK): Implement this  --> DONE
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]
            ih, iw, num_channels = image.shape

            for scale in self.scales:
                # Make black patch
                ph = int(scale * ih)
                pw = int(scale * iw)
                patch = np.full((ph, pw, num_channels), 0, dtype='uint8')

                # Generate random coordinates
                x = np.random.randint(0, (1 - scale) * iw)
                y = np.random.randint(0, (1 - scale) * ih)

                # Cutout step
                image[y : y + ph, x : x + pw, :] = patch

            for i, target in enumerate(targets):
                x1, y1, x2, y2, _ = target
                x1 = int(iw * x1)
                y1 = int(ih * y1)
                x2 = int(iw * x2)
                y2 = int(ih * y2)
                bounded_region = image[y1 : y2, x1 : x2, :]
                bounded_area = (y2 - y1) * (x2 - x1)

                # Count number of non-black pixels in bounding box
                num_black_pixels = cv2.countNonZero(cv2.cvtColor(bounded_region, cv2.COLOR_BGR2GRAY))

                # Remove bounding box if more than half area of bounded region is cutout
                if num_black_pixels / bounded_area < self.p:
                    targets[i][:4] = 0

            return { "image": image, "targets": targets }
        else:
            return sample

class RandomFlip(object):

    def __init__(self, p=0.5):
        # Probability to flip an image
        self.p = p

    def __call__(self, sample):
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]
            # Flip image on y-axis
            image = cv2.flip(image, 1)
            # Transform targets
            targets = targets.copy()
            targets[:, 0], targets[:, 2] = 1 - targets[:, 2], 1 - targets[:, 0]
            return { "image": image, "targets": targets }
        else:
            return sample

class RandomHSV(object):
    def __init__(self, hgain, sgain, vgain, p=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, sample):
        # TODO (TK): Implement this  --> DONE
        if random() < self.p:
            image, targets = sample["image"], sample["targets"]

            # Convert BGR to HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            image[:, :, 0] = (self.hgain * image[:, :, 0]).astype(int) # Changes the H value
            image[:, :, 1] = (self.sgain * image[:, :, 1]).astype(int) # Changes the S value
            image[:, :, 2] = (self.vgain * image[:, :, 2]).astype(int) # Changes the V value

            # Convert HSV to BGR
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            return { "image": image, "targets": targets }
        else:
            return sample

class Resize(object):

    def __init__(self, image_dim):
        # (image_w, image_h)
        self.image_dim = image_dim

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        image = cv2.resize(image, self.image_dim)
        return { "image": image, "targets": targets }

class ToTensor(object):

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255
        targets = torch.from_numpy(targets).float()
        return { "image": image, "targets": targets }


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, targets = sample["image"], sample["targets"]
        transform = transforms.Normalize(self.mean, self.std)
        image = transform(image)
        return { "image": image, "targets": targets }