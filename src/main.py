import argparse
import datetime
import os
import shutil
import yaml

from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import fix_seed
from utils.logging_utils import get_logger
from utils.training_utils import train

SEED = 3407

def main(config, device, writer, logger):
    """
    Main functions. Parses the config file in "prepare_training()", and then
    launches the training loop in "train()".

    Parameters:
        config (dict)           : dict of training configuration parameters as specified in the .yaml file.
        device (str)            : either 'cpu' or 'cuda:<device_no>'.
        writer (SummaryWriter)  : tensorboard writer object for run logging.
        logger (logger.Logger)  : logger object for stout and sterr.
    """
    if logger:
        logger.info('Training session started.')
        print('Training session started.')

    # Start training session
    train(config, device, writer, logger)

    if logger:
        logger.info('Training session ended.')
        print('Training session ended.')
    
    # When done with training, make sure all pending events 
    # have been written to disk and close the tensorboard logger
    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='setup')
    parser.add_argument("--config", type=str, default='/home/thuynh/ms-thesis/data/04_model_input/config/yolov3.yaml', help='Configuration setup file to use for model training.')
    parser.add_argument("--device", type=str, default='cpu', help='Device (CUDA if available) to use for model training.')
    parser.add_argument("--run-tensorboard", type=bool, default=False, choices=[True, False], help='If True, open tensorboard in a browser window at the end of training.')
    parser.add_argument("--verbose", type=bool, default=True, choices=[True, False], help='If False, only logs training progress. If True, also logs the preparation phase.')

    args = parser.parse_args()
    
    try:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as ex:
        print("Exception: ", ex)

    # Make logdir name unique
    try:
        # If running with SLURM, use the SLURM job ID
        logdir_id = os.environ['SLURM_JOB_ID']
    except: 
        # If running locally, use current timestamp
        logdir_id = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
        logdir_id = logdir_id.replace(":", "_")
    
    # The directory where the training outputs will be saved is under:
    # "~/ms-thesis/data/06_model_output/runs"
    # The name structure of the current run folder is the following:
    # 1. Arch_Name: name of object detection architecture (e.g., YOLO, SSD, etc.)
    # 2. Unique_Log_ID: either a timestamp or the SLURM job ID
    # 3. Backbone_Name: name of the backbone network (e.g., ResNet, etc.)
    # 4. Resize: resize resolution (e.g., 144, 256, 512, etc.)
    # 5. Batch_Size: batch size for dataloading (e.g., 1, 8, 16, etc.) 

    logdir = os.path.join(
        "/home/thuynh/ms-thesis/data/06_model_output/runs", 
        config['model']['arch'], # arch_name
        logdir_id + '_' + # unique_log_id
        config['model']['backbone'] + '_' + # backbone_name
        str(config['dataset']['resize']) + '_' + # resize
        'bs' + str(config['training']['batch_size']) # batch_size
    ) 

    # Setup tensorboard writer
    tb_writer = SummaryWriter(log_dir=logdir)

    # Save a copy of the config file in the logging directory
    # This also creates the logdir folder
    shutil.copy(args.config, logdir)

    # Setup logger object
    if args.verbose:
        logger = get_logger(logdir)
    else:
        logger = None

    if logger:
        logger.info(f'Output folder: {logdir}')
        print(f'Output folder: {logdir}')

    # Fix seed
    fix_seed(SEED)

    # Start training
    main(config, args.device, tb_writer, logger)

    # [Optional] Run tensorboard after training is done
    # This will open a browser window
    if args.run_tensorboard:
        os.system(f'tensorboard --logdir {logdir}')