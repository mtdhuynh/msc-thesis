import torch

from torchvision.models.detection import FasterRCNN

from detection_models import build_detection_backbone

def fasterrcnn(backbone='resnet50', pretrained=False, num_classes=7, device=torch.device('cpu'), other_kwargs={'trainable_layers': 5}):
    """
    Loads a FasterRCNN model with the specified backbone. 

    Parameters:
        backbone (str)          : the feature extractor. Usually a CNN model of sorts (e.g., ResNet50).
        pretrained (bool)       : whether to use pretrained weights or not. 
        num_classes (int)       : number of output classes.
        device (torch.device)   : device to load the model on.
        other_kwargs (dict)     : admissible kwargs for backbone, anchor generator, model (pretrained, trainable_layers, etc.).

    Returns:
        model (torch.nn.Module)
    """
    # Separate backbone kwargs from model kwargs
    backbone_kwargs = {k: v for k,v in other_kwargs.items() if k not in FasterRCNN.__init__.__code__.co_varnames}
    model_kwargs = {k: v for k,v in other_kwargs.items() if k in FasterRCNN.__init__.__code__.co_varnames}

    # Get another backbone classifier model and return only its features with the output channels.
    backbone, anchor_generator = build_detection_backbone(backbone, pretrained=pretrained, **backbone_kwargs)

    # Use default AnchorGenerator and default Head
    model = FasterRCNN(
        backbone, 
        num_classes,
        rpn_anchor_generator=anchor_generator, 
        image_mean=(0.0, 0.0, 0.0), # do not normalize
        image_std=(1.0, 1.0, 1.0), # do not normalize
        **model_kwargs
    )

    return model.to(device)