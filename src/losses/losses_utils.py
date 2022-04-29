from losses import YOLOLoss

SUPPORTED_LOSSES = ('yolo')

def get_loss_fn(logger=None, **params):
    """
    This function retrieves an appropriate loss function depending on the model chosen.

    We implement model-dependent losses, as different object detection models output different
    predictions (e.g., one-stage detectors outputs =/= two-stage detectors outputs).

    It returns the selected loss function, which can be called and "backward()" applicable to its outputs.

    Parameters:
        logger (logging.logger)     : python logger object.
        **params (dict)             : admissible custom loss kwargs. 

    Returns:
        loss_fn (torch.nn)
    """
    model_type = params['name']
    params = {k: v for k,v in params.items() if k!='name'}

    try:
        if model_type == 'yolo':
            # Extract the YOLO model
            yolo_model = params['model']
            params = {k: v for k,v in params.items() if k!='model'}

            loss_fn = YOLOLoss(model=yolo_model, **params)
        # more losses and model-support coming.
    except Exception as e:
        print(e)
        logger.info(f'{model_type} loss not yet supported. Available losses: {SUPPORTED_LOSSES}')

    if logger:
        logger.info(f'Using a {model_type} network-compatible loss function.')

    return loss_fn