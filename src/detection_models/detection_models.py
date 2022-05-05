import torch

import detection_models

SUPPORTED_ARCHITECTURES = ['fasterrcnn', 'retinanet', 'yolov3', 'yolov5']

class ODModel():
    """
    Custom model for Object Detection (OD) tasks.
    Given a model name (for architecture and backbone), it tries loading the model from
    the model zoo and return it.
    """
    def __init__(self, params, device):
        """
        Loads one of the models present in models.model_zoo according to the
        arch name provided. 

        Parameters:
            params (dict)           : config dict containing model architecture, backbone, etc.
            device (torch.device)   : device on which to load the model.
        """
        # Double-check device is available
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Get specifications
        self.model_arch = params['arch']
        self.pretrained = params['pretrained']
        self.model_backbone = params['backbone'] # can also be the model version (e.g. yolov3_tiny)
        self.num_classes = params['num_classes']
        # backbone parameters (mainly for torchvision.models, not YOLO-models)
        self.other_kwargs = {k: v for k,v in params.items() if k not in ('arch', 'pretrained', 'backbone', 'num_classes')}

        # Load model from our model_zoo
        try:
            # Get the function from the package list
            model_func = detection_models.__dict__[self.model_arch.lower()]

            # Initialize the model (model will be returned already on selected device)
            self.model = model_func(backbone=self.model_backbone, pretrained=self.pretrained, num_classes=self.num_classes, device=self.device, other_kwargs=self.other_kwargs)
        except Exception as e:
            print(e)
            print(f'Available object detection model architectures: {SUPPORTED_ARCHITECTURES}')