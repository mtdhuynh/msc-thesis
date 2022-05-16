import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator, DefaultBoxGenerator

from detection_models import ODModel

def build_detection_backbone(backbone_name, pretrained=False, ssd=False, **kwargs):
    """
    Loads a specific backbone for object detection models. 

    For a list of all available backbones, look here: https://pytorch.org/vision/stable/models.html#classification.
    
    The backbone is used as a feature extractor and must have an "out_channels" attribute.
    The function:
        1. Loads a backbone (tipically a CNN) from torchvision.models model zoo with the specified arguments (pretrained, trainable_layers, etc.).
        2. Converts the backbone to a feature extractor. 
        3. Defines anchor_sizes and aspect_ratios of the anchors depending on the output feature maps from the backbone.
        
    At the moment, only models with "features" and "features[-1].out_channels" attributes are supported. 
    
    In the future, more models will be supported by using the "torchvision.models.feature_extraction.create_feature_extractor()" function.
    See here how it works: https://pytorch.org/vision/main/feature_extraction.html.
    N.B.: you still need to add the "backbone.out_channels" attribute to the backbone, before inputting it to the object detection model. 

    Returns the backbone model (as a feature extractor), and the AnchorGenerator object with the specified anchor size(s) 
    (according to the number of the backbone's output feature maps) and the correspoding aspect ratios (for each anchor size). 

    Anchor sizes and aspect ratios should be in the following form: Tuple[Tuple[int] * n], with n being the number of output feature maps.

    Backbones without FPN return only one feature map (as a tensor), whereas ones with FPN (ResNet and MobileNet only, currently)
    return up to 5 feature maps.

    Parameters:
        backbone_name (str) : name of the backbone. Available backbones can be found in "torchvision.models".
        pretrained (bool)   : whether to download pretrained weights or not.
        ssd (bool)          : use a DefaultBoxGenerator instead of AnchorGenerator for SSD models as in the paper.
        **kwargs (dict)     : admissible kwargs for the backbone models.

    Returns:
        backbone, anchor_generator (torch.nn.Module, torchvision.models.detection.anchor_utils.AnchorGenerator)
    """
    # Build backbone
    if backbone_name.startswith('resnet'):
        # Load ResNet+FPN backbone
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=pretrained, **kwargs) # trainable_layers=n
        # ResNet with FPN backbone outputs 5 feature maps ['0', '1', '2', '3', 'pool'] 
        # We thus define 5x3 anchor sizes and aspect ratios, one anchor size for each feature map
        anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]) 
        aspect_ratios = ((0.7, 0.85, 1.0, 1.5),) * len(anchor_sizes)
    elif backbone_name.startswith('mobilenet'):
        kwargs['fpn'] = kwargs.get('fpn', True)
        # Load MobileNet backbone
        backbone = torchvision.models.detection.backbone_utils.mobilenet_backbone(backbone_name, pretrained=pretrained, **kwargs) # fpn=True/False, trainable_layers=n, etc.

        if kwargs['fpn']: # default from fasterRCNN
            anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
            aspect_ratios = ((0.7, 0.85, 1.0, 1.5),) * len(anchor_sizes)
        else: # no FPN means only one feature map as output from the backbone
            anchor_sizes = ((32, 64, 128, 256, 512, ), )
            aspect_ratios = ((0.7, 0.85, 1.0, 1.5), )        
    else:
        # Load another backbone with "features" attribute
        feature_extractor = torchvision.models.__dict__[backbone_name](pretrained=pretrained, **kwargs)
        backbone = feature_extractor.features
        backbone.out_channels = backbone[-1].out_channels if hasattr(backbone, 'out_channels') else torchvision.models.detection._utils.retrieve_out_channels(backbone, (512, 512))[0]

        #### TO DO: Implement FPN for general backbones ####

        # Generally speaking, off-the-shelf backbones return a single feature map / tensor, 
        # therefore we only need one anchor_size
        anchor_sizes = ((32, 64, 128, 256, 512, ), )
        aspect_ratios = ((0.7, 0.85, 1.0, 1.5), )
        
    # Define the anchor generator
    if ssd:
        # SSD model uses a DefaultBoxGenerator instead of an AnchorGenerator
        anchor_generator = DefaultBoxGenerator(aspect_ratios)
    else:
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    
    return backbone, anchor_generator

def get_model(params, device, logger=None):
    """
    This function returns a torch.nn.Module model, as specified in the config dict.

    Parameters:
        params (dict)           : object detection model configuration parameters (name, num_classes, etc.).
        device (torch.device)   : device to train on. Can be 'cpu' or 'cuda'.
        logger (logging.logger) : python logger object.

    Returns:
        model (torch.nn.Module)
    """
    model = ODModel(params, device).model

    if not isinstance(logger, (str, type(None))):
        logger.info(f'Loaded {params["arch"]} model with the following parameters: {", ".join([f"{k}={v}" for k,v in params.items()])}.')

    return model