import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target, reduction='mean'):
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)

# def iou_loss(output, target):
#     n, c, h, w = output.shape
#     output  = torch.sigmoid(output)
#     inter = (output*target).sum()
#     union = (output+target).sum()
#     iou  = 1-(inter+1)/(union-inter+1)
#     # return iou 
#     return iou * (n*c*h*w)

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum()
    union = (pred+mask).sum()
    iou  = 1-(inter+1)/(union-inter+1)
    return iou