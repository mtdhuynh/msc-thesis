import torch

AVAILABLE_MODELS = ['yolov3', 'yolov3_spp', 'yolov3_tiny']

def yolov3(backbone='yolov3', pretrained=True, num_classes=6, device=torch.device('cpu')):
    """
    YOLOv3 model architecture.
    Loads a YOLOv3 version from torch.hub. Mainly, the model's implementations
    come from Ultralytics's repo.
    See https://github.com/ultralytics/yolov3/blob/master/models.

    The backbone for ['yolov3', 'yolov3_spp'] is a DarkNet53. For ['yolov3_tiny'] 
    is a custom "tiny" CNN.

    For more torch.hub.load available arguments, see: https://docs.ultralytics.com/tutorials/pytorch-hub/

    Parameters:
        backbone (str)          : which backbone/version to load for the model.
        pretrained (bool)       : whether to download a pretrained version or not.
        num_classes (int)       : number of output classes.
        device (torch.device)   : device to load the model on. 
    """
    # Set autoshape=False for custom training.
    # Autoshape adds a layer for automatic input parsing (cv2, PIL, np, torch.Tensor), but needs to be disabled for training
    # and for custom training/evaluation pipelines. 
    # 
    # See: https://docs.ultralytics.com/tutorials/pytorch-hub/#training
    # See: https://github.com/ultralytics/yolov3/blob/master/hubconf.py

    try:
        return torch.hub.load('ultralytics/yolov3', backbone, pretrained=pretrained, classes=num_classes, device=device, autoshape=False)
    except Exception as e:
        print(f'{e}')
        print(f'Available YOLOv3 models (to specify in "backbone" parameter) in torch.hub: {AVAILABLE_MODELS}. See https://github.com/ultralytics/yolov3/blob/master/models/hub for a list of all available models.')