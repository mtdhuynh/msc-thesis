import argparse
import copy
import datetime
import os
import shutil
import time
import yaml
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

from utilities.general_utils import fix_seed
from utilities.logging_utils import get_logger
from utilities.training_utils import train

SEED = 3407

def main(config, device, scaler, tb_writer, logger=None):
    """
    Launches the main training function.
    """
    # Start training
    if not isinstance(logger, (str, type(None))):
        logger.info('Training session started.')

    train(config, device, scaler, tb_writer, logger)
    
    # End training
    if not isinstance(logger, (str, type(None))):
        logger.info('Training session ended.')
    
    # When done with training, make sure all pending events 
    # have been written to disk and close the tensorboard logger
    tb_writer.flush()
    tb_writer.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='setup')
    parser.add_argument("--config", type=str, default='/home/thuynh/ms-thesis/data/04_model_input/config/yolov3.yaml', help='Configuration setup file to use for model training.')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (CUDA if available) to use for model training.')
    parser.add_argument("--no-amp", action='store_true', help='Disable automatic mixed precision (AMP) training mode.')
    parser.add_argument("--run-tensorboard", action='store_true', help='Open tensorboard in a browser window at the end of training.')
    parser.add_argument("--no-verbose", action='store_true', help='Do not save logs output.')

    args = parser.parse_args()
    
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as ex:
        print("Exception: ", ex)

    # Make logdir name unique
    # Use the timestamp for cronological ordering
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_")
    
    # If running with SLURM, get the JobID (else use 0000)
    slurm_id = os.environ.get('SLURM_JOB_ID', '0000')
    
    # Unique ID is: "timestamp_jobID"
    logdir_id = "_".join([ts, str(slurm_id)])
    
    # The directory where the training outputs will be saved is under:
    # "~/ms-thesis/data/06_model_output/runs"
    # The name structure of the current run folder is the following:
    # 1. Arch_Name: name of object detection architecture (e.g., YOLO, SSD, etc.)
    # 2. Unique_Log_ID: timestamp + the SLURM job ID
    # 3. Backbone_Name: name of the backbone network (e.g., ResNet, etc.)
    # 4. Resize: resize resolution (e.g., 144, 256, 512, etc.)
    # 5. Batch_Size: batch size for dataloading (e.g., 1, 8, 16, etc.) 

    # Example: runs/yolov3/2022-02-23_10_48_32_2542_....

    logdir = os.path.join(
        "/home/thuynh/ms-thesis/data/06_model_output/runs", 
        config['model']['arch'], # arch_name
        "_".join(
            [
                logdir_id, # timestamp_jobID
                config['model']['backbone'], # model backbone
                str(config['dataset']['transforms']['resize']), # input image size (after transform)
                'bs'+str(config['training']['batch_size']) # batch size
            ]
        )
    ) 

    # Setup tensorboard writer
    tb_writer = SummaryWriter(log_dir=logdir)

    # Save a copy of the config file in the logging directory
    # This also creates the logdir folder
    shutil.copy(args.config, logdir)
    os.mkdir(os.path.join(logdir, 'models')) # folder to save models into

    # Setup logger object
    if args.no_verbose:
        logger = logdir
    else:
        logger = get_logger(logdir)

    if not isinstance(logger, (str, type(None))):
        logger.info(f'Output folder: {logdir}')

    # Fix seed
    fix_seed(SEED)
    if not isinstance(logger, (str, type(None))):
        logger.info(f'Fixed random seeds for reproducibility: {SEED}.')

    # Automatic Mixed Precision training
    if args.no_amp or args.device=='cpu': # disable AMP for CPU jobs
        use_amp = False
    else: # use AMP
        use_amp = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # "enabled" argument allows switching between default and mixed precision without if-else

    main(config, args.device, scaler, tb_writer, logger)

    # [Optional] Run tensorboard after training is done
    # This will open a browser window
    if args.run_tensorboard:
        os.system(f'tensorboard --logdir {logdir}')