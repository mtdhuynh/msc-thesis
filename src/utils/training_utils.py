import torch

from data.dataset_utils import get_dataloader, get_dataset, get_transforms
from losses.losses_utils import get_loss_function
from metrics.metrics_utils import get_metrics
from models.models_utils import get_model
from optimizers.optimizers_utils import get_optimizer
from schedulers.schedulers_utils import get_lr_scheduler

def prepare_training(config, device, tb_writer, logger):
    """
    Parse the config dict and load all the training specs, returning them as dict.

    This function does all the heavy lifting: retrieving datasets and dataloaders,
    loading models, optimizers, metrics, etc.

    Parameters:
        config (dict)               : config specs read from a yaml file.
        device (str)                : selected device to run training on. Either 'cpu' or 'cuda:<N>'.
        tb_writer (SummaryWriter)   : tensorboard writer.
        logger (logging.logger)     : python logger object.

    Returns:
        training_dict (dict)
    """
    ##### DEVICE #####
    try:
        device = torch.device(device) # user-specified

        if logger:
            logger.info(f'Using selected device: {device}')
            print(f'Using selected device: {device}')
    except Exception as ex:
        print(f'Specified device not available. Expected ["cpu", "cuda"], got {device}.')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if logger:
            logger.info(f'Using default device: {device}')
            print(f'Using default device: {device}')

    ##### TRANSFORMS #####
    transforms = {
        'train': get_transforms(mode='train', specs=config['dataset']['transforms'], logger=logger),
        'val': get_transforms(mode='val', specs=config['dataset']['transforms'], logger=logger)
    }

    ##### DATASETS #####
    datasets = {
        'train': get_dataset(mode='train', transforms=transforms, fpath=config['dataset']['fpath'], cache=config['dataset']['cache'], tb_writer=tb_writer, logger=logger),
        'val': get_dataset(mode='val', transforms=transforms, fpath=config['dataset']['fpath'], cache=config['dataset']['cache'], logger=logger)
    }

    ##### DATALOADERS #####
    dataloaders = {
        'train': get_dataloader(mode='train', dataset=datasets['train'], specs=config['training'], logger=logger),
        'val': get_dataloader(mode='val', dataset=datasets['val'], specs=config['training'], logger=logger)
    }

    ##### MODEL #####
    model = get_model(specs=config['model'], device=device, logger=logger)

    ##### OPTIMIZER #####
    optimizer = get_optimizer(model=model, specs=config['training']['optimizer'], logger=logger)

    ##### LR SCHEDULER #####
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, specs=config['training']['lr_scheduler'], logger=logger)

    ##### LOSS FUNCTION #####
    loss_function = get_loss_function(specs=config['training']['loss'], logger=logger)

    ##### METRICS #####
    metrics = get_metrics(specs=config['training']['metrics'], logger=logger)

    ##### EPOCHS & SAVING #####
    epochs = config['training']['num_epochs']
    early_stop = config['training']['early_stop']

    if logger:
        logger.info(f'Training for {epochs} epochs.')
        print(f'Training for {epochs} epochs.')

        logger.info(f'Early stop after {early_stop} epochs without improvements in loss function minimization.')
        print(f'Early stop after {early_stop} epochs without improvements in loss function minimization.')

    save_best = config['checkpoints']['save_best'] # True or False
    save_checkpoints = config['checkpoints']['save_checkpoints'] if isinstance(config['checkpoints']['save_checkpoints'], int) else False 
    resume_checkpoint = config['checkpoints']['resume_checkpoint'] 

    ##### PREPARE TRAINING DICT #####
    training_dict = {
        'device': device,
        'model': model,
        'dataloaders': dataloaders,
        'loss_function': loss_function,
        'optimizer': optimizer,
        'metrics': metrics,
        'lr_scheduler': lr_scheduler,
        'epochs': epochs,
        'early_stop': early_stop,
        'save_best': save_best,
        'save_checkpoints': save_checkpoints,
        'resume_checkpoint': resume_checkpoint
    }

    return training_dict