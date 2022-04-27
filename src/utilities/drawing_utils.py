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