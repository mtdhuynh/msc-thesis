from detection_models import ODModel

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

    if logger:
        logger.info(f'Loaded {specs["arch"]} model with the following parameters: {", ".join([f"{k}={v}" for k,v in specs.items()])}.')

    return model

