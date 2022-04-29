import logging
import os
import sys

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
        'Augmented Images': dataset.visualize(n=20, nrow=5, transformed='train', bbox_width=2, font_scale=0.5, thickness=1)
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
    run_id = os.path.basename(logdir).split('_')[0]

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