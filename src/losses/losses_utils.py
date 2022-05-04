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
    if not params:
        # Torchvision models already compute losses internally, so no arguments are passed here
        # and we return None
        loss_fn = None
    elif params['name'] == 'yolo':
        try:
            # Extract the YOLO model
            model_type = params['name']
            yolo_model = params['model']
            params = {k: v for k,v in params.items() if k not in ('name', 'model')}

            loss_fn = YOLOLoss(model=yolo_model, **params)
        except Exception as e:
            print(e)
            logger.info(f'{model_type} loss not yet supported. Available losses: {SUPPORTED_LOSSES}')

    if not isinstance(logger, str):
        logger.info(f'Using {loss_fn} loss function.')

    return loss_fn