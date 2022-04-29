"""
Codebase adapted from YOLOv5 by Ultralytics (https://github.com/ultralytics/yolov5/) - GPL-3.0 License.

Loss functions.
"""

from enum import auto
import torch

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Computes the bboxes IoU. 
    IoU returned can either be:
    - Generalized IoU: https://arxiv.org/pdf/1902.09630.pdf.
    - Distance IoU: https://arxiv.org/abs/1911.08287v1.
    - Complete IoU: https://arxiv.org/abs/1911.08287v1.
    """
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def smooth_BCE(eps=0.1):
    """
    Implements label smoothing as in: https://arxiv.org/pdf/1902.04103.pdf [EQN 3].

    See: https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    It returns the label smoothing values for positive and negative targets, respectively.

    Parameters:
        eps (float) : smoothing value.
    
    Returns:
        positive_smoothing, negative_smoothing (float, float)
    """
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(torch.nn.Module):
    """
    FocalLoss implementation, specifically for YOLO networks outputs.

    It wraps the focal loss around a BCEWithLogitsLoss(reduction='none') and computes the sigmoid_focal_loss.
    Very similar to the torchvision.ops implementation: https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html#sigmoid_focal_loss.

    Example:
    >>> loss_fn = FocalLoss(torch.nn.BCEWithLogitsLoss(), gamma=1.5)    
    """
    def __init__(self, loss_fn=torch.nn.BCEWithLogitsLoss(), gamma=1.5, alpha=0.25):
        """
        Loads the focal loss class.

        The BCEWithLogitsLoss reduction argument must be set to 'none', in order to apply
        the FocalLoss to each element. 
        N.B.: the FocalLoss reduction argument can still be 'mean' or 'sum', not necessarily 'none'. 
        It will use the default from the wrapped loss function. 

        Parameters:
            loss_fn (torch.nn)  : wrapped loss function. Must be torch.nn.BCEWithLogitsLoss().
            gamma (float)       : focal loss gamma value.
            alpha (float)       : focal loss alpha value.
        """
        # Initialize
        super().__init__()
        self.loss_fn = loss_fn  # must be torch.nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        # Use default wrapped loss's reduction, but set to 'none' locally for FL computation.
        self.reduction = loss_fn.reduction
        self.loss_fn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, targets):
        """
        Computes the focal loss between predicted outputs and ground-truth targets.
        Follows the TensorFlow implementation: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/focal_loss.py

        Parameters:
            pred (torch.Tensor)     : predicted outputs from a YOLO network.
            targets (torch.Tensor)  : ground-truth targets (bounding boxes), as tensors.

        Returns:
            loss (torch.Tensor)
        """
        # Compute BCEWithLogits loss
        loss = self.loss_fn(pred, targets)

        # Get the probabilities from the logits
        pred_prob = torch.sigmoid(pred)
        
        # Compute the Focal Loss
        p_t = targets * pred_prob + (1 - targets) * (1 - pred_prob)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class YOLOLoss():
    """
    Custom loss for YOLO networks. 
    Adaptation of https://github.com/ultralytics/yolov5/blob/master/utils/loss.py, with hard-coded hyperparameters. 

    Uses either a binary-cross entropy loss (with logits) or focal loss.
    """
    sort_obj_iou = False

    def __init__(self, model, fl_gamma=0.0, autobalance=False):
        """
        Define the loss for YOLO networks.

        Parameters:
            model (torch.nn.Module) : the YOLO network. Needed for some hyperparameters.
            fl_gamma (float)        : focal loss gamma value. If 0.0, then BCE loss is used instead.
            autobalance (bool)
        """
        self.model = model
        self.fl_gamma = fl_gamma
        self.autobalance = autobalance
        
        # Get device where model is
        device = next(self.model.parameters()).device

        # Use BCE loss by default
        class_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        objectness_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        if self.fl_gamma > 0:
            # Use Focal Loss
            class_loss = FocalLoss(class_loss, self.fl_gamma)
            objectness_loss = FocalLoss(objectness_loss, self.fl_gamma)


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

        # Get the Detect module parameters (Detect module is the last block of a YOLO network)
        self.detect_module = self.model.model[-1]

        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.detect_module.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(self.detect_module.stride).index(16) if self.autobalance else 0  # stride 16 index
        self.class_loss, self.objectness_loss, self.gr = class_loss, objectness_loss, 1.0

        # The following are defined at YOLO model definition 
        self.na = self.detect_module.na  # number of anchors
        self.nc = self.detect_module.nc  # number of classes
        self.nl = self.detect_module.nl  # number of layers
        self.anchors = self.detect_module.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """
        Compute YOLO Loss. 

        It returns the bbox, class and objectness losses summed together (as one torch.Tensor value) multiplied by the batch size,
        and the separate losses as a torch.Tensor((bbox_loss, objectness_loss, class_loss)).

        "targets" should contain the following: 
        [0, class, normalized(x-center, y-center, w, h)]
        with shape: [6].

        Parameters: 
            p (torch.Tensor)        : predicted outputs from a YOLO network.
            targets (torch.Tensor)  : ground-truth bboxes as tensor (with bbox class).

        Returns:
            batch_loss, loss_items (torch.Tensor, torch.Tensor)
        """
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # Compute losses for each layer predictions 
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.class_loss(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.objectness_loss(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Builds a list of targets-predictions combinations to compute the losses for.
        Decides which anchors to assign to each annotations. 

        "targets" should contain the following: 
        [0, class, normalized(x-center, y-center, w, h)]
        with shape: [6].

        It returns the list of assigned anchors-to-targets with corresponding classes.

        Parameters: 
            p (torch.Tensor)        : predicted outputs from a YOLO network.
            targets (torch.Tensor)  : ground-truth bboxes as tensor (with bbox class). See docstring.

        Returns:
            tcls, tbox, indices, anch (list, list, list, list)
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < 4.0  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch