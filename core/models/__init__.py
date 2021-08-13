from .classification.efficientnet.efficientnetb0 import EfficientnetB0
from .classification.resnet.resnet18 import Resnet18
from .classification.darknet.darknet53 import Darknet53
from .detection.ssd.ssd300 import SSD300
from .detection.ssd_lite.ssdlite300 import SSDLite300
from .detection.yolov3.yolov3 import YoloV3

models = {
    "detection": {
        "ssd300": SSD300,
        "ssdlite300": SSDLite300,
        "yolov3": YoloV3
    },
    "classification": {
        "efficientnetb0": EfficientnetB0,
        "resnet18": Resnet18,
        "darknet53": Darknet53
    }
}
