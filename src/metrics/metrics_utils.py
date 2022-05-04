from torchmetrics.detection.mean_ap import MeanAveragePrecision

def get_metrics(device='cpu', logger=None, **params):
    """
    This function retrieves the mean Average Precision (mAP) metric used in object detection.
    It relies mainly on the torchmetrics.MeanAveragePrecision implementation.
    Se: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html

    N.B.: the torchmetrics implementation asks as input the bbox coordinates format ('xyxy', 'xywh', 'cxcywh').
    Make sure the output predictions are in one of those formats, else the metric computation won't happen.

    Parameters:
        device (torch.device)   : should be the same device as where the input data resides.
        logger (logging.logger) : python logger object.
        **params (dict)         : admissible torchmetrics.MeanAveragePrecision kwargs (metrics configuration parameters - name, device, etc.).

    Returns:
        metrics (list)
    """
    metrics = MeanAveragePrecision(**params).to(device)

    if not isinstance(logger, str):
        logger.info(f'Evaluating the model against mAP metric.')

    return metrics