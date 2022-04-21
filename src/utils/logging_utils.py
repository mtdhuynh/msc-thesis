import logging
import os

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
    run_id = os.path.basename(logdir).split('_')[0]

    file_path = os.path.join(logdir, f"run_{run_id}.log")
    hdlr = logging.FileHandler(file_path)

    formatter = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s")
    hdlr.setFormatter(formatter)

    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return logger