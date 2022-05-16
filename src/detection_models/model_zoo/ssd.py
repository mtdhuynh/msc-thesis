import torch
import torchvision
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD, _vgg_extractor

from detection_models import build_detection_backbone

def ssd(backbone='vgg16', pretrained=False, num_classes=7, device=torch.device('cpu'), other_kwargs={'trainable_layers': 5}):
    """
    Loads an SSD model with the specified backbone. 

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
    backbone_kwargs = {k: v for k,v in other_kwargs.items() if k not in SSD.__init__.__code__.co_varnames}
    model_kwargs = {k: v for k,v in other_kwargs.items() if k in SSD.__init__.__code__.co_varnames}

    if backbone=='vgg16':
        size = 512 if model_kwargs['size'][0] == 512 else 300
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]] if size!=512 else [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]]
        scales = [0.51, 0.69, 0.87, 1.05, 1.13, 1.35, 1.52] if size!=512 else [0.51, 0.69, 0.87, 1.05, 1.13, 1.35, 1.52, 1.97]
        # Use custom backbones more appropriate for SSD
        backbone = torchvision.models.vgg16(pretrained=pretrained)
        backbone = _vgg_extractor(backbone, True if size==512 else False, backbone_kwargs.get("trainable_layers", 5))
        anchor_generator = DefaultBoxGenerator(aspect_ratios, scales=scales)
    else:
        # Get another backbone classifier model and return only its features with the output channels.
        backbone, anchor_generator = build_detection_backbone(backbone, pretrained=pretrained, ssd=True, **backbone_kwargs)

        # SSD expects out_channels as a list.
        backbone.out_channels = [backbone.out_channels]
        anchor_generator.aspect_ratios = (anchor_generator.aspect_ratios[0], )

    # Use default AnchorGenerator and default Head
    model = SSD(
        backbone, 
        num_classes=num_classes,
        anchor_generator=anchor_generator, 
        image_mean=(0.0, 0.0, 0.0), # do not normalize
        image_std=(1.0, 1.0, 1.0), # do not normalize
        **model_kwargs
    )

    return model.to(device)