from .cvat_xml import CVATXMLDataset
from .yolov4 import YOLOv4Dataset
from .coco_json import COCOJSONDataset
from torchvision.datasets import ImageFolder

readers = {
    "cvat_xml": CVATXMLDataset,
    "yolov4": YOLOv4Dataset,
    "coco_json": COCOJSONDataset,
    "image_folder": ImageFolder,
}
