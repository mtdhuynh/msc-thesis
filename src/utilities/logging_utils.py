import logging
import os
import sys

import torch

def log_tb_images(dataset, tb_writer):
    """
    Generates grids of original and (training) augmented images with their
    corresponding bounding boxes drawn, and then logs them onto tensorboard.
    
    Parameters:
        dataset (ODDataset)                     : an ODDataset(torch.utils.data.Dataset) object.
        tb_writer (tensorboard.SummaryWriter)   : the tensorboard writer object.
    """
    # Generate grids
    image_grids = {
        'Original Images': dataset.visualize(n=20, nrow=5),
        'Augmented Images': dataset.visualize(n=20, nrow=5, transformed=True, bbox_width=2, font_scale=0.5, thickness=1)
    }

    # Log images to tensorboard
    for tag, grid in image_grids.items():
        tb_writer.add_image(tag, grid)

def get_logger(logdir):
    """
    Create logger object to store training session info.

    Parameters:
        logdir (str)    : full path to the current run log directory
    """
    logger = logging.getLogger("Automatic Polyp Detection")
    
    # Get unique ID of run
    run_id = os.environ.get('SLURM_JOB_ID', os.path.basename(logdir).split('_')[0]) # either jobID or timestamp

    file_path = os.path.join(logdir, f"run_{run_id}.log")

    # Log output to file
    file_handler = logging.FileHandler(file_path)

    formatter = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log output to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    logger.setLevel(logging.INFO)

    return logger

def save_model(fname, model, epoch, optimizer, lr_scheduler, run_history, best_loss):
    """
    Saves the input model, optimizer, scheduler's state dicts. 

    Parameters:
        fname (str)                             : filename (with path) to save. 
        model (torch.nn.Module)                 : trained model.
        epoch (int)                             : current training epoch (+1). 
        optimizer (torch.optim)                 : optimizer.
        lr_scheduler (torch.optim.lr_scheduler) : learning rate scheduler.
        run_history (dict)                      : dict with the logged metrics, losses, etc.
        best_loss (float)                       : best loss value so far.

    Returns: 
        None
    """
    # Make sure the device is on the CPU for compatibility-safe loading on any system
    model.to('cpu')

    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'run_history': run_history,
            'best_loss': best_loss
        },
        fname # .pt extension
    )