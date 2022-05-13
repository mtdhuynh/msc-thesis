import albumentations as A
import cv2
import torchvision

def draw_bbox(img, bboxes, labels, bbox_width=2, bbox_colors=[(0,100,10)], text_color=(255,255,255), font_scale=1.2, thickness=2):
    """
    Draws all the input bounding boxes on the input image.

    The input image has to be a np.ndarray with HWC format. 
    Bounding boxes must be supplied as a list of coordinates list: [coords1, coords2, etc.].
    Bounding box coordinates must have the following format: [xmin, ymin, xmax, ymax].

    Parameters:
        img (np.ndarray)    : an image as array (torch.Tensor, numpy.ndarray, PIL.Image, etc.).
        bboxes (list)       : list of bbox coordinates lists. bbox should have the following format: [xmin, ymin, xmax, ymax]
        labels (list)       : list of corresponding label names (str).
        bbox_width (int)    : width of the bounding box outline.
        bbox_colors (list)  : list of (R,G,B) color tuples for the corresponding bounding boxes outline color.
        text_color (tuple)  : (R,G,B) color tuple for the label text color.
        font_scale (float)  : font size with respect to the specific font base size.
        thickness (int)     : thickness of line used for text drawing.

    Returns:
        img (np.ndarray)
    """

    for bbox, label, color in zip(bboxes, labels, bbox_colors):
        # Extract coordinates
        x_min, y_min, x_max, y_max = bbox

        # Plot bounding box on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, bbox_width)
        
        # Draw the label text
        font_scale = font_scale
        ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Make sure that the text box is inside the image
        text_x_min = x_min
        text_y_min = y_min - int(1.5 * text_height)
        text_x_max = x_min + text_width
        text_y_max = y_min

        if text_y_min < 0:
            text_y_max = y_min - (text_y_min)
            text_y_min = y_min + bbox_width-1
        if text_x_max > img.shape[-2]:
            text_x_min = x_min - (text_x_max - x_max)
            text_x_max = x_max

        cv2.rectangle(img, (text_x_min - bbox_width+1, text_y_min), (text_x_max, text_y_max), color, -1)

        cv2.putText(
            img,
            text=label,
            org=(text_x_min, text_y_max - int(0.35 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, 
            color=text_color,
            thickness=thickness, 
            lineType=cv2.LINE_AA,
        )

    return img

def plot_grid(images, nrow=4, padding=1, normalize=False):
    """
    Plot images into a grid. 
    The input list of images can strictly contain torch.Tensor images with CHW format.

    Internally, it uses torchvision.utils.make_grid().

    N.B.: the fastest way to obtain the images list is to store cv2.imread()-read images
    (np.array) in a list, and then convert them to torch.Tensor and CHW format with the 
    albumentations.pytorch.ToTensorV2 transform.

    Example:
    > tensorize = ToTensorV2()
    > tensor_images = [tensorize(np_img) for np_img in np_images]

    The np_images list can contain any np.array image that has been processed
    in any way (e.g., using cv2 to draw bounding boxes on them).

    Parameters:
        images (list)   : list of torch.Tensor images with CHW format.
        nrow (int)      : number of images per row in the grid.
        padding (int)   : spacing (in pixel) between adjacent images.
        normalize (bool): whether the tensors are normalized or not.

    Returns:
        grid (torch.Tensor)
    """
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize)

    return grid

def normalize_bbox(bbox, src_format='xywh', img_width=512, img_height=512):
    """
    Normalizes an input bbox coordinate by image dimensions. 
    Accepts both 'xyxy' and 'xywh' formats (i.e., pascal_voc or coco formats),
    where 'xyxy'=[x1,y1,x2,y2] and 'xywh'=[x-center,y-center,widht,height].

    Adapted from: https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/bbox_utils.py#L59.

    If format is 'xyxy', it is the same as converting Pascal-VOC bboxes to albumentations format.
    If format is 'xywh', it is the same as converting COCO bboxes to YOLO format.

    Parameters:
        bbox (tuple)        : bbox coordinates. Length of bbox should be 4.
        src_format (str)    : bbox coordinates format. Either 'xyxy' or 'xywh'.
        img_width (int)     : source image width.
        img_height (int)    : source image height.

    Returns:
        normalized_bbox (list)
    """
    if src_format == 'xyxy': # pascal-voc to albumentations
        norm_bbox = A.normalize_bbox(bbox, img_height, img_width)
    elif src_format == 'xywh': # coco to yolo
        norm_x_center = bbox[0] / img_width # x-center
        norm_y_center = bbox[1] / img_height # y-center
        norm_width = bbox[2] / img_width # bbox-width
        norm_height = bbox[3] / img_height # bbox-height

        norm_bbox = (norm_x_center, norm_y_center, norm_width, norm_height)
    else:
        raise Exception(f'Unknown bbox format: {src_format}. Supported formats: ["xyxy", "xywh"].')

    return list(norm_bbox)

def denormalize_bbox(bbox, src_format='xywh', img_width=512, img_height=512):
    """
    Denormalizes an input bbox coordinate by image dimensions. 
    It is the inverse of 'normalize_bbox()'.

    Input bbox coordinates should be within [0, 1].

    Accepts both 'xyxy' and 'xywh' formats (i.e., albumentations or yolo formats),
    where 'xyxy'=normalized([x1,y1,x2,y2]) and 'xywh'=normalized([x-center,y-center,widht,height]).

    Adapted from: https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/bbox_utils.py#L59.

    If format is 'xyxy', it is the same as converting albumentations bboxes to Pascal-VOC format.
    If format is 'xywh', it is the same as converting YOLO bboxes to COCO format.

    Parameters:
        bbox (tuple)        : bbox coordinates. Length of bbox should be 4.
        src_format (str)    : bbox coordinates format. Either 'xyxy' or 'xywh'.
        img_width (int)     : source image width.
        img_height (int)    : source image height.

    Returns:
        normalized_bbox (list)
    """
    if src_format == 'xyxy': # pascal-voc to albumentations
        norm_bbox = A.denormalize_bbox(bbox, img_height, img_width)
    elif src_format == 'xywh': # coco to yolo
        norm_x_center = bbox[0] * img_width # x-center
        norm_y_center = bbox[1] * img_height # y-center
        norm_width = bbox[2] * img_width # bbox-width
        norm_height = bbox[3] * img_height # bbox-height

        norm_bbox = (norm_x_center, norm_y_center, norm_width, norm_height)
    else:
        raise Exception(f'Unknown bbox format: {format}. Supported formats: ["xyxy", "xywh"].')

    return list(norm_bbox)

def resize_bbox(bboxes, old_height, old_width, new_height, new_width):
    """
    Resizes input bboxes from current image height and width to
    provided new image dimensions. 

    Parameters:
        bboxes (list[np.array]) : list of bounding boxes coordinates. Shape should be [N, 4].
        old_height (int)        : current height of the image the bboxes belong to.
        old_width (int)         : current width of the image the bboxes belong to.
        new_height (int)        : height to resize the bboxes to.
        new_width (int)         : width to resize the bboxes to.

    Returns:
        bboxes (list[np.array])
    """
    height_ratio = new_height / old_height
    width_ratio = new_width / old_width

    bboxes[:, 0] = bboxes[:, 0] * width_ratio
    bboxes[:, 2] = bboxes[:, 2] * width_ratio
    bboxes[:, 1] = bboxes[:, 1] * height_ratio
    bboxes[:, 3] = bboxes[:, 3] * height_ratio

    return bboxes
