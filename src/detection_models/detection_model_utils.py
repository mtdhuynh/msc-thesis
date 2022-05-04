import torchvision

from detection_models import ODModel

def build_detection_backbone(backbone_name, pretrained=False,  **kwargs):
    """
    Loads a specific backbone for object detection models. 

    For a list of all available backbones, look here: https://pytorch.org/vision/stable/models.html#classification.
    
    The backbone is used as a feature extractor and must have an "out_channels" attribute.
    The function:
        1. Loads a backbone (tipically a CNN) from torchvision.models model zoo with the specified arguments (pretrained, trainable_layers, etc.).
        2. Converts the backbone to a feature extractor. 
        
    At the moment, only models with "features" and "features[-1].out_channels" attributes are supported. 
    
    In the future, more models will be supported by using the "torchvision.models.feature_extraction.create_feature_extractor()" function.
    See here how it works: https://pytorch.org/vision/main/feature_extraction.html.
    N.B.: you still need to add the "backbone.out_channels" attribute to the backbone, before inputting it to the object detection model. 

    Parameters:
        backbone_name (str) : name of the backbone. Available backbones can be found in "torchvision.models".
        pretrained (bool)   : whether to download pretrained weights or not.
        **kwargs (dict)     : admissible kwargs for the backbone models.

    Returns:
        backbone (torch.nn.Module)
    """
    # Build backbone
    if backbone_name.startswith('resnet'):
        # Load ResNet+FPN backbone
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=pretrained, **kwargs) # trainable_layers=n
    elif backbone_name.startswith('mobilenet'):
        # Load MobileNet backbone
        backbone = torchvision.models.detection.backbone_utils.mobilenet_backbone(backbone_name, pretrained=pretrained, **kwargs) # fpn=True/False, trainable_layers=n, etc.
    else:
        # Load another backbone with "features" attribute
        try:
            # Not all backbones have a "features" attribute, so fail gracefully
            feature_extractor = torchvision.models.__dict__[backbone_name](pretrained=pretrained, **kwargs)
            backbone = feature_extractor.features
            backbone.out_channels = backbone[-1].out_channels # not every last features block has "out_channels"
        except Exception as e:
            raise Exception(f'Encountered unknown error: {e}. If your selected backbone is not available, you can try one of the default backbones: ["resnet<[18,50,101,152]>", "mobilenet_v<[2,3_small,3_large]>"].')

    return backbone

def get_model(specs, device, logger=None):
    """
    This function returns a torch.nn.Module model, as specified in the config dict.

    Parameters:
        specs (dict)            : object detection model configuration parameters (name, num_classes, etc.).
        device (torch.device)   : device to train on. Can be 'cpu' or 'cuda'.
        logger (logging.logger) : python logger object.

    Returns:
        model (torch.nn.Module)
    """
    model = ODModel(specs, device).model

    if not isinstance(logger, str):
        logger.info(f'Loaded {specs["arch"]} model with the following parameters: {", ".join([f"{k}={v}" for k,v in specs.items()])}.')

    return model