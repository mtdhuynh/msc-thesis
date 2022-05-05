import torch

from torchvision.models.detection import RetinaNet

from detection_models import build_detection_backbone

def retinanet(backbone='resnet50', pretrained=False, num_classes=7, device=torch.device('cpu'), other_kwargs={'trainable_layers': 5}):
    """
    Loads a RetinaNet model with the specified backbone. 
    
    The default model is a RetinaNet model with a ResNet50+FPN backbone.

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
    backbone_kwargs = {k: v for k,v in other_kwargs.items() if k not in ('img_size')}

    # Get another backbone classifier model and return only its features with the output channels.
    backbone, anchor_generator = build_detection_backbone(backbone, pretrained=pretrained, **backbone_kwargs)

    # Use default AnchorGenerator and default Head
    model = RetinaNet(
        backbone, 
        num_classes, 
        anchor_generator=anchor_generator,
        min_size=other_kwargs['img_size'], # do not resize
        max_size=other_kwargs['img_size'], # do not resize
        image_mean=(0.0, 0.0, 0.0), # do not normalize
        image_std=(1.0, 1.0, 1.0) # do not normalize
    )

    return model.to(device)
