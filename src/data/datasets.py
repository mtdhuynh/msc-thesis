import json
import os

import cv2
import numpy as np
import torch
import tqdm

from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
    
class ODDataset(torch.utils.data.Dataset):
    """
    Custom dataset object for Object Detection (OD) tasks.
    Reads and stores the images, corresponding bounding boxes and labels.
    """
    def __init__(self, fpath, transforms=None, cache=False):
        """
        Reads images and labels pairs, given the path to the dataset split (train or val).

        Users can supply a list of transformations to apply to the image. If none are provided, these
        will only include scaling, normalization, and conversion to torch.Tensor.
        Transforms should be a list of "albumentations.Compose" transforms.

        Th "cache" option allows users to read and load data to memory when instantiating the ODDataset object. 
        Caching takes some time and requires appropriate hardware resources, as it loads and stores all the images 
        on CPU first (then eventually batches are moved to the GPU if available, through "to(device)"). 
        In turn, this should allow for speed-up in training times, as we reduce the I/O bottleneck of loading and
        moving images from CPU to GPU at runtime. 
        
        Parameters:
            fpath (str)         : full path to the directory that contains 'images' and 'labels' folders.
            transforms (list)   : list of transforms to apply to the images. 
            cache (bool)        : whether to cache the data or not. 
        """    
        if os.path.isdir(fpath):
            self.images_path = os.path.join(fpath, 'images')
            self.labels_path = os.path.join(fpath, 'labels')
        else:
            raise Exception(f'The specified data folder path is not valid: {fpath}.')
        
        # Get images and labels paths list
        self.images_list = os.listdir(self.images_path)
        self.labels_list = [fname[:-4]+'.json' for fname in self.images_list]
        
        # Transforms
        # Should be 'albumentations.Compose" transforms.
        # "torch.Compose" transforms are not yet supported. 
        if transforms.__class__ == Compose([None]).__class__:
            self.transforms = transforms
        else:
            raise Exception(f'Transforms should be provided. Supported transforms types: [{Compose([None]).__class__}]. Got: {transforms.__class__}')
        
        # Caching
        self.cache = cache
        
        if self.cache:            
            print('Caching data...')
            self.cached_data = []
            
            for img_fname, label_fname in tqdm.tqdm(zip(self.images_list, self.labels_list), total=len(self.images_list)):
                image, label = self.read_and_transform(img_fname, label_fname, self.transforms)
                    
                # Append to cache list as tuple
                self.cached_data.append((image, label))
            
        # Also read polyp classes
        self.polyp_classes_file = '/home/thuynh/ms-thesis/data/02_intermediate/polyp_classes.json'
        with open(self.polyp_classes_file, 'r') as f:
            self.polyp_classes = json.load(f)['polyp_classes']

    def __getitem__(self, index):
        if self.cache: # If caching, we retrieve the image-label pairs from the cached list
            image, label = self.cached_data[index]
        else: # Else, we perform loading on request
            img_fname = self.images_list[index]
            label_fname = self.labels_list[index]
            
            image, label = self.read_and_transform(img_fname, label_fname, self.transforms)
        
        return image, label
        
    def __len__(self):
        return len(self.images_list)

    def __repr__(self):
        return f'ODDataset(fpath="{os.path.split(self.images_path)[0]}", transforms={self.transforms}, cache={self.cache})'\
        
    def _read(self, img_fname, label_fname):
        """
        Takes image and annotation paths and loads them as numpy array and dict, respectively.

        Images are read as np.ndarray with RGB, HWC format. 
        Labels are read as dict.

        Parameters:
            img_fname (str)     : filename of the image.
            label_fname (str)   : filename of the annotation file.
        
        Returns:
            img (np.ndarray)
            label (dict)
        """
        # Load image and convert to RGB
        image = cv2.imread(os.path.join(self.images_path, img_fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label
        with open(os.path.join(self.labels_path, label_fname), 'r') as f:
            label = json.load(f)

        return image, label

    def _transform(self, image, label, transforms):
        """
        Applies an augmentation pipeline to both input image and label.
        Uses albumentations library to do so.

        The parameters for the transforms are defined in the "bbox_params"
        argument of albumentations.Compose(). 
        See https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/.

        For images without annotations (negative images), it still returns the
        target dict. However, in this case, its keys/values are empty tensors.
        See https://github.com/pytorch/vision/releases/tag/v0.6.0.

        It returns a dict (the output of applying the transforms to image and label)
        containing the following keys: ['image', 'bboxes', 'label_ids', 'label_names'].

        Parameters:
            image (np.ndarray)                  : input image (RGB, HWC formats).
            label (dict)                        : annotation dict as specified in the annotation template.
            transforms (albumentations.Compose) : list of augmentations to apply.

        Returns:
            transformed (dict)
        """
        # Parse the labels
        # For each annotation of each image, save bbox coordinates, class IDs and names.
        if label['labels']: # positive images
            bboxes = [lbl['bbox'] for lbl in label['labels']]
            class_ids = [lbl['category_id'] for lbl in label['labels']]
            class_names = [lbl['category_name'] for lbl in label['labels']]
        else: # negative images
            # For images without annotations, use empty tensors
            bboxes = torch.zeros((0,4), dtype=torch.float32)
            class_ids = torch.zeros(0, dtype=torch.int64)
            class_names = torch.zeros(0, dtype=torch.int64)

        if transforms is None:
            transforms = self.transforms

        transformed = transforms(image=image, bboxes=bboxes, label_ids=class_ids, label_names=class_names)

        return transformed

        
    def read_and_transform(self, img_fname, label_fname, transforms=None):
        """
        Reads and applies augmentations to image and labels simultaneously. 
        Uses cv2 and albumentations libraries. 

        It returns a torch.Tensor (with CHW format), and a dict containing 
        the following keys: ['image_id', 'boxes', 'label_ids', 'label_names'].

        Parameters:
            img_fname (str)                     : filename of the image.
            label_fname (str)                   : filename of the annotation file.
            transforms (albumentations.Compose) : list of transformations to apply.
        
        Returns:
            transformed_image (torch.Tensor)
            target (dict)
        """
        # Load image and label
        image, label = self._read(img_fname, label_fname)
        
        # Apply transformation pipeline
        if transforms is None:
            transforms = self.transforms

        transformed = self._transform(image, label, transforms)

        # Augmented image
        transformed_image = transformed['image']

        # Augmented labels
        target = {
            'image_id': img_fname,
            'boxes': torch.as_tensor(transformed['bboxes'], dtype=torch.float32), # Bbox coordinates
            'label_ids': torch.as_tensor(transformed['label_ids'], dtype=torch.int64), # Corresponding class IDs
            'label_names': transformed['label_names'] # Corresponding class names
        }

        return transformed_image, target