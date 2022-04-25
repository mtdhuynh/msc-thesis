import torch

AVAILABLE_MODELS = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x', 'yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']

def YOLOv5(backbone='yolov5s', pretrained=True, num_classes=6, device=torch.device('cpu')):
    """
    YOLOv5 model architecture.
    Loads a YOLOv5 version from torch.hub. Mainly, the model's implementations
    come from Ultralytics's repo.
    See https://github.com/ultralytics/yolov3.

    Versions released from ultralytics are:
    ['n', 's', 'm', 'l', 'x', 'n6', 's6', 'm6', 'l6', 'x6'].

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
    # See: https://github.com/ultralytics/yolov5/blob/master/hubconf.py
    
    try:
        return torch.hub.load('ultralytics/yolov5', backbone, pretrained=pretrained, classes=num_classes, device=device, autoshape=False)
    except Exception as e:
        print(f'{e}')
        print(f'Available YOLOv5 models (to specify in "backbone" parameter) in torch.hub: {AVAILABLE_MODELS}.')