import logging
import os
import sys

import psutil
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

def get_timestamp(seconds):
    """
    Return string with time formatted in days, hours, minutes, seconds:
    '<dd>d <hh>h <mm>m <ss>s'

    N.B.: Code will break if time exceeds 31 days.

    Parameters:
        seconds (int)   : time to convert in [s].
    
    Returns:
        timestamp (str)
    """
    
    minutes, secs = divmod(int(seconds), 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    
    return f'{days:d}d {hrs:02d}h {mins:02d}m {secs:02d}s'

def save_model(fname, model_state_dict, epoch, optimizer_state_dict, lr_scheduler_state_dict, run_history, best_metrics, best_loss):
    """
    Saves the input model, optimizer, scheduler's state dicts. 

    You should make sure to move the model to CPU before saving, for
    compatibility with any system (GPU-able or not).

    Parameters:
        fname (str)                     : filename (with path) to save. 
        model_sate_dict (dict)          : trained model state_dict.
        epoch (int)                     : current training epoch (+1). 
        optimizer_state_dict (dict)     : optimizer state_dict.
        lr_scheduler_state_dict (dict)  : learning rate scheduler state_dict.
        run_history (dict)              : dict with the logged metrics, losses, etc.
        best_metrics (dict)             : dict with the best mAPs.
        best_loss (float)               : best loss value so far.

    Returns: 
        None
    """
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'run_history': run_history,
            'best_metrics': best_metrics,
            'best_loss': best_loss
        },
        fname # .pt extension
    )

def get_memory_info(used, total):
    """
    Returns input used and total memory (RAM, GPU, etc.) in human-readable format. 

    Expects as input the used and total memory in bytes.

    Parameters:
        used (int)  : amount of used memory in bytes.
        total (int) : total amount of memory in bytes.
    
    Returns:
        (str)

    Example:
    >>> import psutil
    >>> # Get RAM usage in bytes
    >>> used, total = psutil.virtual_memory().used, psutil.virtual_memory().total
    >>> # Print in human-readable format
    >>> print(f"RAM Usage: {get_memory_info(used, total)}") # "<RAM used>GB / <RAM total>GB (<RAM used percentage>%)"
    """    
    return f"{psutil._common.bytes2human(used)}B / {psutil._common.bytes2human(total)}B ({round(used/total*100, 1)}%)"