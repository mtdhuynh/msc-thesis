import os

import albumentations as A

from albumentations.pytorch import ToTensorV2

from data.dataloaders import ODDataloader
from data.datasets import ODDataset
from utilities.logging_utils import log_tb_images

def get_dataloader(mode, dataset, specs, logger=None):
    """
    This functions returns a torch.utils.data.DataLoader object that contains
    the batches of data used for training/evaluation of the model.

    Parameters:
        mode (str)                          : specifies the data split. Either 'train' or 'val'.
        dataset (torch.utils.data.Dataset)  : the input data from which to load the batches.
        specs (dict)                        : dict of dataloading configuration (batch size, number of workers, etc.).
        logger (logging.logger)             : python logger object.

    Returns:
        dataloader (torch.utils.data.DataLoader)
    """
    # Shuffle images at each epoch's start only for training set
    shuffle = True if mode=='train' else False

    # Create torch.utils.data.DataLoader object
    dataloader = ODDataloader(
        dataset, 
        batch_size=specs['batch_size'], 
        shuffle=shuffle,
        num_workers=specs['num_workers'],
        pin_memory=specs['pin_memory']
    )
    
    if logger:
        logger.info(f'Using {mode} dataloader with batch_size={specs["batch_size"]}, shuffle={shuffle}, num_workers={specs["num_workers"]}, pin_memory={specs["pin_memory"]}.')

    return dataloader

def get_dataset(mode, fpath, transforms, cache=False, tb_writer=None, logger=None):
    """
    This function returns a torch.utils.data.Dataset object that contains the
    data used for training/evaluation of the model.

    Parameters:
        mode (str)                              : specifies the data split. Either 'train' or 'val'.
        fpath (str)                             : the full path to the input data folder. Should contain 'train' and 'val' folders.
        transforms (albumentations.Compose)     : list of transforms to be applied for augmentation.
        cache (bool)                            : whether to cache the data when instantiating the dataset object.
        tb_writer (tensorboard.SummaryWriter)   : tensorboard writer.
        logger (logging.logger)                 : python logger object.

    Returns:
        dataset (torch.utils.data.Dataset)
    """
    # Get datasplit path
    data_folder = os.path.join(fpath, mode)

    if logger:
        logger.info(f'Reading input data from: {data_folder}.')

    # Create torch.utils.data.Dataset object
    dataset = ODDataset(fpath=data_folder, transforms=transforms, cache=cache)

    # Log images to tensorboard
    if tb_writer:
        log_tb_images(dataset, tb_writer)

    if logger:
        logger.info(f'Using {len(dataset)} {mode} images.')

    return dataset

def get_transforms(mode, specs, normalize=True, logger=None):
    """
    Load standard transforms based on data split.
    Base transform pipeline is: [Resize, Normalize, ToTensor].
    
    For mode='val', only use the base transforms for reproducibility.
    For mode='train', also add random image transforms to avoid overfitting.

    Supported transforms are from the albumentations library. 
    https://albumentations.ai/docs/api_reference/augmentations/transforms/

    Training transforms are hard-coded and are the ones used in the literature.
    For example, see: https://www.sciencedirect.com/science/article/pii/S0016510720346551, 
    where they use [Vertical/Horizontal Flipping, Sharpen, Brightness].

    N.B.: albumentations applies the transforms also to the corresponding bboxes, and 
    the extent and application of the transforms to these can be fine-tuned using the
    'bbox_params' argument of albumentations.Compose.
    See https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/.

    Parameters:
        mode (str)              : specifies the data split. Either 'train' or 'val'.
        specs (dict)            : dict with specifications for transforms (resize, etc.).
        normalize (bool)        : whether to apply normalization or not. Used mainly for drawing purposes. See ODDataset.visualize() method.
        logger (logging.logger) : python logger object.
    
    Returns:
        transforms (albumentations.Compose)
    """
    # Read specifications
    resize = specs['resize'] # Output resize resolution
    min_area = specs['min_area'] # Bboxes smaller than "min_area" are dropped
    min_visibility = specs['min_visibility'] # Bboxes smaller than "min_visibility*original_bbox_area" are dropped
    
    # Define bbox parameters
    bbox_params = A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=min_visibility, label_fields=['label_ids', 'label_names'])

    # Define base transform pipeline
    if normalize:    
        BASE_TRANSFORMS = [
                A.Resize(resize, resize),
                A.Normalize(), # use COCO mean and std
                ToTensorV2()
            ]
    else: # Remove Normalize and ToTensor because we want np.uint8 and HWC format
        BASE_TRANSFORMS = [
                A.Resize(resize, resize)
            ]

    # Define transforms based on mode
    if mode=='val':
        transforms = A.Compose(BASE_TRANSFORMS, bbox_params=bbox_params)
    elif mode=='train':
        transforms = A.Compose(
            [
                A.Flip(p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.RandomScale(scale_limit=0.15, p=0.5),
                A.OneOf(
                    [
                        A.GaussianBlur(p=0.25),
                        A.MotionBlur(p=0.25),
                        A.GaussNoise(p=0.25),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.1)
                    ]
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=0.4),
                        A.ColorJitter(hue=0, p=0.3)
                    ]
                )
            ] + BASE_TRANSFORMS,
            bbox_params=bbox_params
        )

    if logger:
        logger.info(f'For {mode} data, using the following transforms: {transforms}.')

    return transforms

