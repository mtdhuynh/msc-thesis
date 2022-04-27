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
        self.transforms = transforms
        
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

    def visualize(self, transformed=False, n=10, nrow=5, bbox_width=5, font_scale=1.2, thickness=2):
        """
        Visualizes a grid of images with bounding box annotations.

        It loads random images and labels, draws the bounding boxes and corresponding label texts,
        then puts them neatly in a grid.
        Uses torch.utils.make_grid().

        If transformed=True, it displays the images and bounding boxes after applying
        training augmentations. 

        Recommended font_scale, thickness, and bbox_width values are:
        - font_scale=1.2 + thickness=2 + bbox_width=5 (all defaults) --- if transformed=False (i.e., 1000x1000 images).
        - font_scale=0.5 + thickness=1 + bbox_width=2                --- if transformed=True (e.g., 512x512 images).

        It returns a grid of images, which is in the form of one torch.Tensor image with CHW format.
        
        Parameters:
            transformed (bool)  : whether to apply augmentations before displaying or not.
            n (int)             : number of images to visualize.
            nrow (int)          : number of rows for the image grid.
            bbox_width (int)    : controls the thickness of the bbox outline (in pixel).
            font_scale (float)  : controls the size of the label text (relative to the font base size).
            thickness (int)     : controls the thickness of the line used for drawing the text (in pixel).

        Returns:
            grid (torch.Tensor)
        """
        # Import here to avoid conflicts       
        from data.dataset_utils import get_transforms
        from utilities.drawing_utils import draw_bbox, plot_grid 
        
        # Get n random images and labels
        RNG = np.random.default_rng()

        indexes = RNG.choice(len(self.images_list), n, replace=False)

        image_samples = [self.images_list[idx] for idx in indexes]
        label_samples = [self.labels_list[idx] for idx in indexes]

        images = []

        # Load n images and labels
        for img_fname, lbl_fname in zip(image_samples, label_samples):
            image, label = self._read(img_fname, lbl_fname)

            if not transformed:
                if not label['labels']: 
                    # If no annotations (negative image), plot it anyway and go to next image
                    images.append(image)
                    continue
                elif label['labels']: 
                # If annotations, draw on original and positive images
                # Extract relevant annotations (we assume one annotation per image)
                    bbox_coords = [lbl['bbox'] for lbl in label['labels']]
                    label_ids = [lbl['category_id'] for lbl in label['labels']]
                    label_names = [lbl['category_name'] for lbl in label['labels']]
            else: # Apply transforms to image
                # Get training transforms without normalization
                transforms_no_norm = get_transforms(mode='train', specs={'resize': 512, 'min_area': 900, 'min_visibility': 0.2}, normalize=False)
                # Apply transforms
                transf = self._transform(image, label, transforms_no_norm)

                # Process output as needed for draw_bbox() -- mainly convert to np.array
                image = transf['image']
                bbox_coords = np.array(transf['bboxes'], dtype=np.int64)
                label_names = transf['label_names']
                label_ids = np.array(transf['label_ids'], dtype=np.int64)

            # Get bbox color from template
            bbox_colors = [eval(polyp['outline']) for polyp in self.polyp_classes for id in label_ids if polyp['id']==id]

            # Draw bbox
            img_with_bbox = draw_bbox(image, bbox_coords, label_names, bbox_width=bbox_width, bbox_colors=bbox_colors, font_scale=font_scale, thickness=thickness)

            images.append(img_with_bbox)

        # Organize images in a grid
        # To use torchvision.utils.make_grid() we need to convert the images to tensors
        # We also resize them, as not all of the images have equal size
        tensorize = Compose([Resize(1000,1000), ToTensorV2()])

        images = [tensorize(image=img)['image'] for img in images]

        # Generate grid
        grid = plot_grid(images, nrow=nrow, padding=1, normalize=False)

        return grid