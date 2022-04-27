from detection_models.detection_models import ODModel
from detection_models.model_utils import get_model

from detection_models.model_zoo.yolov3 import YOLOv3
from detection_models.model_zoo.yolov5 import YOLOv5

__all__ = [
    get_model,
    ODModel,
    YOLOv3,
    YOLOv5
]