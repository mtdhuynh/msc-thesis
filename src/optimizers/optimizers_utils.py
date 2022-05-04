from optimizers import ODOptimizer

def get_optimizer(model, logger=None, **params):
    """
    This function returns an optimizer, as specified in the config dict.

    Parameters:
        model (torch.nn.Module)     : the selected model for object detection.
        logger (logging.logger)     : python logger object.
        **params (dict)             : admissible torch.optim kwargs (optimizer configuration parameters - name of optimization algorithm, lr, etc.).

    Returns:
        optimizer (torch.optim)
    """
    # Get the name of the optimizer
    name = params['name']
    # Get the rest of the parameters (to use .pop('name') we would need to make a deepcopy)
    params = {k: v for k,v in params.items() if k!='name'} 
    
    optimizer = ODOptimizer(model, name, **params).optimizer

    if not isinstance(logger, str):
        logger.info(f'Using optimizer: {optimizer}.')

    return optimizer