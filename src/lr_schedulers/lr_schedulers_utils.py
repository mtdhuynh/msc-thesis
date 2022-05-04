from lr_schedulers import Constant_LR, LR_Scheduler

def get_lr_scheduler(optimizer, logger=None, **params):
    """
    This function returns a learning rate scheduler, as specified in the config dict.

    Parameters:
        optimizer (torch.optim) : optimizer object with the chosen optimization algorithm.
        logger (logging.logger) : python logger object.
        **params (dict)         : admissible torch.optim.lr_scheduler kwargs (learning rate configuration parameters - initial/final lr, scheduler type, etc.).

    Returns:
        lr_scheduler (torch.optim.lr_scheduler)
    """
    # If no scheduler parameters are specified, uses constant lr (default from optimizer)
    if not params:
        lr_scheduler = Constant_LR(optimizer)

        if not isinstance(logger, (str, type(None))):
            logger.info(f'No learning rate scheduler specified. Using default constant learning rate: {lr_scheduler}')
    else: # Else, load a scheduler from torch.optim.lr_scheduler
        if not isinstance(logger, (str, type(None))):
            logger.info(f'Using the following learning rate scheduler: {", ".join([f"{k}: {v}" for k,v in params.items()])}.')

        name = params['name']
        params = {k: v for k,v in params.items() if k!='name'}

        lr_scheduler = LR_Scheduler(optimizer, name, **params).lr_scheduler

    return lr_scheduler