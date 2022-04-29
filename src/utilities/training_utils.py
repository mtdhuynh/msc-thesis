from calendar import c
import torch

from detection_models import get_model
from data import get_dataloader, get_dataset, get_transforms
from losses import get_loss_fn
from metrics import get_metrics
from optimizers import get_optimizer
from lr_schedulers import get_lr_scheduler

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
    except Exception as ex:
        # Select available device automatically
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if logger:
            logger.info(f'Specified device not available. Expected ["cpu", "cuda"], got {device}.')

            logger.info(f'Using default device: {device}')

    ##### TRANSFORMS #####
    normalize = config['dataset']['transforms'].pop('normalize')

    transforms = {
        'train': get_transforms(mode='train', params=config['dataset'], normalize=normalize, logger=logger),
        'val': get_transforms(mode='val', params=config['dataset'], normalize=normalize, logger=logger)
    }

    ##### DATASETS #####
    datasets = {
        'train': get_dataset(mode='train', transforms=transforms, fpath=config['dataset']['fpath'], bbox_format=config['dataset']['format'], cache=config['dataset']['cache'], tb_writer=tb_writer, logger=logger),
        'val': get_dataset(mode='val', transforms=transforms, fpath=config['dataset']['fpath'], bbox_format=config['dataset']['format'], cache=config['dataset']['cache'], logger=logger)
    }

    ##### DATALOADERS #####
    dataloaders = {
        'train': get_dataloader(mode='train', dataset=datasets['train'], params=config['training'], logger=logger),
        'val': get_dataloader(mode='val', dataset=datasets['val'], params=config['training'], logger=logger)
    }

    ##### MODEL #####
    model = get_model(specs=config['model'], device=device, logger=logger)

    ##### OPTIMIZER #####
    optimizer = get_optimizer(model=model, logger=logger, **config['training']['optimizer'])

    ##### LR SCHEDULER #####
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, logger=logger, **config['training']['lr_scheduler'])

    ##### LOSS FUNCTION #####
    # YOLOLoss needs also the model as input
    if config['model']['arch'].startswith('YOLO'):
        config['training']['loss']['model'] = model 

    loss_function = get_loss_fn(logger=logger, **config['training']['loss'])

    ##### METRICS #####
    metrics = get_metrics(device=device, logger=logger, **config['training']['mAP'])

    ##### EPOCHS & SAVING #####
    epochs = config['training']['num_epochs']
    early_stop = config['training']['early_stop']

    save_best = config['checkpoints']['save_best'] # True or False
    save_checkpoints = config['checkpoints']['save_checkpoints']
    resume_checkpoint = config['checkpoints']['resume_checkpoint']
    
    if logger:
        logger.info(f'Training for {epochs} epochs.')
        logger.info(f'Early stop after {early_stop} epochs without improvements in loss function minimization.')
        if save_best:
            logger.info('Checkpointing every best model.')
        if save_checkpoints:
            logger.info('Checkpointing every 50 epochs.')
        if resume_checkpoint:
            logger.info('Resuming training from checkpoint: {resume_checkpoint}.')
        else:
            logger.info('Training from scratch.')

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