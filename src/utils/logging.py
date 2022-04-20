import datetime
import logging
import os

from torchvision import utils

def log_images(dataloaders, writer):
    """
    Load a batch of images, draw the corresponding ground-truth bounding boxes
    and finally log the images to tensorboard.
    
    Parameters:
    dataloaders (dict)                  : a dict containing torch.data.DataLoader objects.
    writer (tensorboard.SummaryWriter)  : the tensorboard writer object.
    """
    # Font for drawing bounding boxes (required to edit font size)
    custom_font = '/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf'

    if not os.path.exists(custom_font):
        custom_font = None # revert to defaul behaviour if font does not exists 

    # Load a batch of images from training sets
    images, labels = next(iter(dataloaders['train']))
    
    for idx, lbl in enumerate(labels):
        # Draw bounding box if available
        if lbl['boxes']:
            images[idx] = utils.draw_bounding_boxes(
                images[idx], 
                [lbl['boxes']], # torch.Tensor of shape [N, 4]
                [lbl['labels']], # torch.Tensor of shape [N]
                width=5,
                font=custom_font,
                font_size=30
            )

    # Make a grid of images from the batch
    # "normalize=True" denormalizes images (they have been transformed during dataloading)
    img_grid = utils.make_grid(images, normalize=True)

    # Log images to tensorboard
    writer.add_image("Example Images", img_grid)

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