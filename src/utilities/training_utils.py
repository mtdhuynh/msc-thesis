import copy
import os
import sys
import time

from collections import defaultdict, OrderedDict
import psutil

import torch
import tqdm

from detection_models import get_model
from data import get_dataloader, get_dataset, get_transforms
from losses import get_loss_fn
from metrics import get_metrics
from optimizers import get_optimizer
from lr_schedulers import get_lr_scheduler
from utilities.logging_utils import get_timestamp, get_memory_info, save_model

def prepare_training(config, _device, tb_writer, logger):
    """
    Parse the config dict and load all the training specs, returning them as dict.

    This function does all the heavy lifting: retrieving datasets and dataloaders,
    loading models, optimizers, metrics, etc.

    Parameters:
        config (dict)               : config specs read from a yaml file.
        _device (str)                : selected device to run training on. Either 'cpu' or 'cuda[:<N>]'.
        tb_writer (SummaryWriter)   : tensorboard writer.
        logger (logging.logger)     : python logger object.

    Returns:
        training_dict (dict)
    """
    ##### DEVICE #####
    device = torch.device(_device if torch.cuda.is_available() else 'cpu')

    if not isinstance(logger, (str, type(None))):
        if not torch.cuda.is_available() and _device.startswith('cuda'):
            logger.info(f'Selected {_device} device not available. Using default device: {device}')
        else:
            logger.info(f'Using device: {device}')

    ##### TRANSFORMS #####
    normalize = config['dataset']['transforms'].pop('normalize')
    bbox_format = config['dataset']['format']

    transforms = {
        'train': get_transforms(mode='train', params=config['dataset'], normalize=normalize, logger=logger),
        'val': get_transforms(mode='val', params=config['dataset'], normalize=normalize, logger=logger)
    }

    ##### DATASETS #####
    datasets = {
        'train': get_dataset(mode='train', transforms=transforms['train'], fpath=config['dataset']['fpath'], bbox_format=config['dataset']['format'], cache=config['dataset']['cache'], tb_writer=tb_writer, logger=logger),
        'val': get_dataset(mode='val', transforms=transforms['val'], fpath=config['dataset']['fpath'], bbox_format=config['dataset']['format'], cache=config['dataset']['cache'], logger=logger)
    }

    ##### DATALOADERS #####
    dataloaders = {
        'train': get_dataloader(mode='train', dataset=datasets['train'], params=config['training'], logger=logger),
        'val': get_dataloader(mode='val', dataset=datasets['val'], params=config['training'], logger=logger)
    }

    ##### MODEL #####
    if config['model']['arch'] not in ('yolov3', 'yolov5'):
         # torchvision OD models implement internally a Normalize+Resize transform.
         # We have our own augmentation pipeline, and by supplying our own resize size, we prevent a further resizing.
         # YOLO models do not have this internal transform.
        config['model']['img_size'] = config['dataset']['transforms']['resize']
    model = get_model(specs=config['model'], device=device, logger=logger)

    ##### OPTIMIZER #####
    optimizer = get_optimizer(model=model, logger=logger, **config['training']['optimizer'])

    ##### LR SCHEDULER #####
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, logger=logger, **config['training']['lr_scheduler'])

    ##### LOSS FUNCTION #####
    # Torchvision models compute the loss internally, therefore we pass None as the loss_fn
    if config['training']['loss'] == 'None':
        loss_fn = get_loss_fn(logger=logger) # returns None
    # YOLOLoss needs also the model as input
    elif config['model']['arch'] in ('yolov3', 'yolov5'):
        config['training']['loss']['model'] = model 

        loss_fn = get_loss_fn(logger=logger, **config['training']['loss'])

    ##### METRICS #####
    # We only use mAP from torchmetrics
    metrics = {'mAP': get_metrics(device=device, logger=logger, **config['training']['mAP'])}

    ##### EPOCHS & SAVING #####
    epochs = config['training']['num_epochs']
    early_stop = config['training']['early_stop']

    save_checkpoints = config['checkpoints']['save_checkpoints']
    # YAML reads None as string, so convert to NoneType if "None", else use the path provided
    resume_checkpoint = None if config['checkpoints']['resume_checkpoint'] == 'None' else config['checkpoints']['resume_checkpoint']
    
    if not isinstance(logger, (str, type(None))):
        logger.info(f'Training for {epochs} epochs.')
        logger.info(f'Early stop after {early_stop} epochs without improvements in loss function minimization.')
        if save_checkpoints:
            logger.info('Checkpointing every 50 epochs.')

    ##### PREPARE TRAINING DICT #####
    training_dict = {
        'device': device,
        'model': model,
        'dataloaders': dataloaders,
        'datasets': datasets,
        'bbox_format': bbox_format,
        'loss_fn': loss_fn,
        'optimizer': optimizer,
        'metrics': metrics,
        'lr_scheduler': lr_scheduler,
        'epochs': epochs,
        'early_stop': early_stop,
        'save_checkpoints': save_checkpoints,
        'resume_checkpoint': resume_checkpoint
    }

    return training_dict

def train(config, device, tb_writer, logger):
    """"
    Main training function. The function takes as input the configuration
    file, the selected device (if available), the tensorboard writer and 
    the logger objects.

    Internally, it parses the config dict and loads all specified utilities
    for training (dataloaders, models, optimizers, etc.).

    Finally, it starts the training loop by calling the internal function "train_one_epoch".
    
    At the end (either early checkout or end of epochs) saves
    best model, metrics, losses, and optimizer's state dicts in a .pt object in
    the corresponding logging dir.

    Parameters:
        config (dict)               : config specs read from a yaml file.
        device (str)                : selected device to run training on. Either 'cpu' or 'cuda:<N>'.
        tb_writer (SummaryWriter)   : tensorboard writer.
        logger (logging.logger)     : python logger object.
    """
    # Logging folder
    if not isinstance(logger, (str, type(None))):
        logdir = logger.handlers[0].baseFilename.rsplit('/', 1)[0]
    elif isinstance(logger, (str, type(None))):
        # if no_verbose was selected, logger is the run output directory
        logdir = logger

    # Read and load specifications from config
    training_dict = prepare_training(config, device, tb_writer, logger)

    # Unpack everything from the training dict
    # Model
    model = training_dict['model']
    device = training_dict['device']
    # Data
    dataloaders = training_dict['dataloaders']
    datasets = training_dict['datasets']
    bbox_format = training_dict['bbox_format']
    # Training utils
    lr_scheduler = training_dict['lr_scheduler']
    loss_fn = training_dict['loss_fn']
    optimizer = training_dict['optimizer']
    metrics = training_dict['metrics']
    # Other
    epochs = training_dict['epochs']
    early_stop = training_dict['early_stop']
    save_checkpoints = training_dict['save_checkpoints']
    resume_checkpoint = training_dict['resume_checkpoint']

    if resume_checkpoint is None: # from scratch
        start_epoch = 0
        run_history = defaultdict(list)
        best_loss = torch.tensor([[float('inf')], [float('inf')], [float('inf')]], dtype=torch.float32).to(device) # [[global loss], [class loss], [reg loss]]
        if not isinstance(logger, (str, type(None))):
            logger.info('Training model from scratch.')
    else:
        # Resume training from checkpoint specified in "resume_checkpoint" (str, fpath to checkpoint)
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            # Load model state dict (checkpointed model must be the same as the one currently loaded)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            # Load optimizer and scheduler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            # Load other utils
            start_epoch = checkpoint['epoch']
            run_history = checkpoint['run_history']
            best_map = checkpoint['best_map']
            best_loss = checkpoint['best_loss'].to(device)
            if not isinstance(logger, (str, type(None))):
                logger.info(f'Resuming training from the following checkpoint: {checkpoint}.')
        except Exception as e:
            print(e)
            if not isinstance(logger, (str, type(None))):
                logger.info(f'Encountered unknown error while loading checkpoint: {e}.')
                logger.info(f'Check the provided checkpoint: {resume_checkpoint}.')
            # Exit
            return None

    # Track training history
    # Copy state_dict to CPU to avoid GPU consumption
    best_model = OrderedDict({k: v.to('cpu') for k, v in model.state_dict().items()})    
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_lr_scheduler = copy.deepcopy(lr_scheduler.state_dict())
    patience = 0

    # Reset the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()

    # Wrap everything in a try-except to fail gracefully without losing progresses
    try:
        for epoch in range(start_epoch, epochs):
            if not isinstance(logger, (str, type(None))):
                logger.info(f'{"-"*15} Epoch {epoch+1}/{epochs} {"-"*15}')
            
            ###### TRAINING ######
            for phase in ('train', 'val'):
                tic = time.time()

                # Tracking
                best_epoch = False
                
                # Reset loss
                epoch_loss = torch.zeros((3,1), dtype=torch.float32).to(device)
                
                # Use training mode also in val, as we want to compute the val loss too
                model.train()
                
                # Progress bar
                pbar = tqdm.tqdm(enumerate(dataloaders[phase]), dynamic_ncols=True, file=sys.stdout)

                for i, (inputs, targets) in pbar:
                    # pbar.set_description(f'[{phase}] Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloaders[phase])}')
                    # Reset the optimizer
                    optimizer.zero_grad(set_to_none=True)

                    inputs = inputs.to(device, non_blocking=True)
                    targets = [{k: v.to(device, non_blocking=True) for k,v in target.items() if k not in ('label_names', 'image_name')} for target in targets]

                    # if not isinstance(logger, (str, type(None))):
                    #     logger.info(f'Memory Summary: {torch.cuda.memory_summary()}')
                    #     logger.info(f'MemGetInfo: {torch.cuda.mem_get_info()}')
                    #     logger.info(f'Memory Currently Allocated: {torch.cuda.memory_allocated()} | Max Allocated Memory: {torch.cuda.max_memory_allocated()}')
                    #     logger.info(f'Memory Currently Reserved: {torch.cuda.memory_reserved()} | Max Reserved Memory: {torch.cuda.max_memory_reserved()}')

                    # Enable gradient computation only for training phase
                    with torch.set_grad_enabled(phase=='train'):
                        # Torchvision models loss
                        if loss_fn is None:
                            # Output from torchvision models is a dict of classification and regression losses
                            loss_dict = model(inputs, targets)
                            losses = sum(loss for loss in loss_dict.values())
                        # ################################################################## YOLO TBD #########################################################
                        # # YOLOLoss
                        # else: # bbox_format=='yolo':
                        #     # Transform labels to a tensor of shape [N, 6], with [image_idx, class_id, [bbox_coords]] content
                        #     # TO DO: move it inside the dataset __getitem__ method
                        #     num_labels = len(targets['boxes']) # how many bbox for each image
                        #     labels_yolo = torch.zeros(num_labels,6)
                        #     if num_labels:
                        #         labels_yolo[:, 1:2] = targets['labels']
                        #         labels_yolo[:, 2:] = targets['boxes']
                            
                        #     labels_yolo.to(device, non_blocking=True)

                        #     # YOLO forward pass
                        #     outputs = model(inputs)
                        #     # Compute YOLOLoss
                        #     losses, loss_dict = loss_fn(outputs, labels_yolo) # loss already multiplied per batch size
                        ########################################################################################################################################

                        # Backward pass in train mode only
                        if phase == 'train':
                            losses.backward()
                            optimizer.step()
                            # Update LR at each batch (only for ['CyclicLR', 'CosineAnnealingLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts']). Other LRs are updated at each epoch
                            if str(lr_scheduler.__class__.__name__) in ('CyclicLR', 'CosineAnnealingLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts'):
                                tb_writer.add_scalar('LR Scheduler', lr_scheduler.get_last_lr()[0], len(dataloaders[phase])*epoch + i+1)
                                run_history['lr_scheduler'].append(lr_scheduler.get_last_lr()[0])
                                lr_scheduler.step()
                        
                    # Accumulate loss per-batch and de-average by multiplying by bs: https://discuss.pytorch.org/t/confused-about-set-grad-enabled/38417/4
                    epoch_loss[0] += (losses.item() * inputs.size(0))
                    epoch_loss[1] += (loss_dict['classification'] * inputs.size(0))
                    epoch_loss[2] += (loss_dict['bbox_regression'] * inputs.size(0))

                    # Update pbar
                    ram_usage = psutil.virtual_memory()
                    pbar.set_description(f'[{phase}] Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloaders[phase])} | Average Batch Global Loss: {losses.item()} | Class Loss: {loss_dict["classification"]} | Bbox Loss: {loss_dict["bbox_regression"]} | RAM Usage: {get_memory_info(ram_usage.used, ram_usage.total)} | GPU Usage: Allocated: {get_memory_info(torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device))}, Reserved: {get_memory_info(torch.cuda.memory_reserved(device), torch.cuda.max_memory_reserved(device))} (Total: {psutil._common.bytes2human(torch.cuda.get_device_properties(device).total_memory)}B)')
                    
                    # if not isinstance(logger, (str, type(None))): # per batch
                    #     logger.info(f'[{phase}] Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloaders[phase])} | Average Batch Loss: {losses.item()} | Class Loss: {loss_dict["classification"]} | Bbox Loss: {loss_dict["bbox_regression"]}')

                # Averaged loss over the epoch
                epoch_loss = epoch_loss / len(datasets[phase])
                
                # Log training epoch results
                epoch_time = get_timestamp(time.time()-tic)[3:]
                
                if not isinstance(logger, (str, type(None))): # per epoch
                    logger.info(f'[{phase}] Epoch #{epoch+1} | Loss: {epoch_loss[0].item()} | Class Loss: {epoch_loss[1].item()} | Bbox Loss {epoch_loss[2].item()} | Time Elapsed: {epoch_time}')

                tb_writer.add_scalar(f'epoch_loss/{phase}', epoch_loss[0].item(), epoch+1)
                tb_writer.add_scalar(f'class_loss/{phase}', epoch_loss[1].item(), epoch+1)
                tb_writer.add_scalar(f'bbox_loss/{phase}', epoch_loss[2].item(), epoch+1)
                run_history[f'epoch_loss/{phase}'].append(epoch_loss[0].item())
                run_history[f'class_loss/{phase}'].append(epoch_loss[1].item())
                run_history[f'bbox_loss/{phase}'].append(epoch_loss[2].item())

                # Best model tracking
                if phase == 'val':
                    if epoch_loss[0] < best_loss[0]:
                        best_loss = copy.deepcopy(epoch_loss)
                        best_epoch = True
                    else:
                        patience += 1
                        # Early checkout
                        if patience >= early_stop:
                            total_time = get_timestamp(time.time() - start_time)
                            if not isinstance(logger, (str, type(None))):
                                logger.info(f'{patience} epochs without loss improvements. Stopping training...')
                                logger.info(f'Training completed after {epoch+1} epochs in {total_time}.')

                            # Save best model
                            save_model(
                                os.path.join(logdir, 'models', f'best_model.pt'),
                                best_model,
                                epoch+1,
                                best_optimizer,
                                best_lr_scheduler,
                                best_run_history,
                                best_map,
                                best_loss
                            )
                            return None # exit
                        else:
                            if not isinstance(logger, (str, type(None))):
                                logger.info(f'{patience} epochs without improvements. Continuing training...')
                    
                    # Checkpointing every 50 epochs
                    if save_checkpoints and (epoch+1) % 50 == 0:
                        if not isinstance(logger, (str, type(None))):
                            logger.info(f'Saving checkpoint @ epoch {epoch+1}...')
                        save_model(
                                os.path.join(logdir, 'models', f'checkpoint_{epoch+1}.pt'),
                                model.state_dict(), # model @ epoch
                                epoch+1,
                                optimizer.state_dict(), # optim @ epoch
                                lr_scheduler.state_dict(), # lr_sched @ epoch
                                run_history,
                                epoch_map,
                                epoch_loss
                            )

            # Update LR at each epoch for other LRs
            if not str(lr_scheduler.__class__.__name__) in ('CyclicLR', 'CosineAnnealingLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts'):
                tb_writer.add_scalar('LR Scheduler', lr_scheduler.get_last_lr()[0], epoch+1)
                run_history['lr_scheduler'].append(lr_scheduler.get_last_lr()[0])
                lr_scheduler.step()

            ###### METRICS (VALIDATION) ######
            tac = time.time()
            epoch_map = {}
            
            # Initialize metrics (see docs: https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html#module-metrics)
            # and reset each epoch
            for metric in metrics.keys():
                metrics[metric].reset()

            phase = 'val'
            model.eval()

            # Progress bar
            pbar = tqdm.tqdm(enumerate(dataloaders[phase]), dynamic_ncols=True, file=sys.stdout)

            for i, (inputs, targets) in pbar:
                pbar.set_description(f'[{phase}] Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(dataloaders[phase])} | Computing mAP metrics in inference mode...')

                inputs = inputs.to(device, non_blocking=True)
                targets = [{k: v.to(device, non_blocking=True) for k,v in target.items() if k not in ('label_names', 'image_name')} for target in targets]

                # Inference
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)

                # Accumulate mAP metrics per batch
                for metric in metrics.keys():
                    metrics[metric].update(outputs, targets)

            # Compute mAP metrics over all batches
            for metric in metrics.keys():
                epoch_map[metric] = metrics[metric].compute() # returns Dict[str: Tensor]

            # Log validation epoch results
            val_time = get_timestamp(time.time() - tac)[3:]

            if not isinstance(logger, (str, type(None))): # per epoch
                logger.info(f'[{phase}] Epoch #{epoch+1} | {" | ".join([f"{k}: {v}" for metric in epoch_map.keys() for k,v in epoch_map[metric].items()])} | Time Elapsed: {val_time}')
            
            for metric in epoch_map.keys():
                for k, v in epoch_map[metric].items():
                    tb_writer.add_scalar(f'{k}', v, epoch+1)
                    run_history[k].append(v)

            # Track best model and metrics
            if best_epoch:
                best_model = OrderedDict({k: v.to('cpu') for k, v in model.state_dict().items()}) # best model || move to CPU to avoid GPU overhead
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                best_lr_scheduler = copy.deepcopy(lr_scheduler.state_dict())
                best_run_history = copy.deepcopy(run_history)
                best_map = copy.deepcopy(epoch_map)
                patience = 0
                if not isinstance(logger, (str, type(None))):
                    logger.info('New best model!')
            # ---- End of epoch ----

        ###### END OF TRAIN/VAL ######
        # Log results
        if not isinstance(logger, (str, type(None))):
            total_time = get_timestamp(time.time() - start_time)
            logger.info(f'Model did not converge in {epochs} epochs.')
            logger.info(f'Training completed after {epochs} epochs in {total_time}.')

        # Save best model so far
        save_model(
            os.path.join(logdir, 'models', f'best_model.pt'),
            best_model,
            epochs,
            best_optimizer,
            best_lr_scheduler,
            best_run_history,
            best_map,
            best_loss
        )
        # save hparams
        tb_writer.add_hparams(
            {
                'batch_size': config['training']['batch_size'],
                'input_size': config['dataset']['transforms']['resize'],
                **config['model'], # model
                **config['training']['optimizer'], # optimizer
                **config['training']['lr_scheduler'], # lr-scheduler
                'early_stop': config['training']['early_stop']
            },
            best_map # TO DO: use best_epoch_map
        )

        return None
    except KeyboardInterrupt:
        # If user wants to interrupt training at any moment,
        # make sure we don't lose any improvement between checkpoints or any progress so far
        if not isinstance(logger, (str, type(None))):
            logger.info(f'Training interruption requested. Saving best model so far and stopping training @ epoch {epoch+1}...')
        
        # save best model so far
        save_model(
            os.path.join(logdir, 'models', f'best_model_{epoch+1}_kbi.pt'),
            best_model,
            epoch+1,
            best_optimizer,
            best_lr_scheduler,
            best_run_history,
            best_map,
            best_loss
        )

        # Log to tensorboard
        tb_writer.add_hparams(
            {
                'batch_size': config['training']['batch_size'],
                'input_size': config['dataset']['transforms']['resize'],
                **config['model'], # model
                **config['training']['optimizer'], # optimizer
                **config['training']['lr_scheduler'], # lr-scheduler
                'early_stop': config['training']['early_stop']
            },
            best_map
        )
        
        if not isinstance(logger, (str, type(None))):
            time_elapsed = get_timestamp(time.time() - start_time)
            logger.info(f'Training stopped after {epoch+1}. Runtime: {time_elapsed}.')