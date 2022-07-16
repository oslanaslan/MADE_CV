import torch
import torch.functional as F
from collections import Counter

# TODO TIP: Using Dice coeff is OK for measuring segmentation quality
# TODO TIP: Optimizing the Dice Loss usually helps segmentation a lot.

def dice_coeff(input, target):
    raise NotImplementedError


def dice_loss(input, target):
    # input = input.to('cuda:0')
    # target = target.to('cuda:0')
    smooth = 1.
    # iflat = input.view(-1)
    # tflat = target.view(-1)
    input_soft = input
    target_one_hot = target
    intersection = torch.sum(input_soft * target_one_hot)
    cardinality = torch.sum(input_soft + target_one_hot)

    dice_score = 2. * intersection / (cardinality + smooth)
    res = torch.mean(1. - dice_score)

    return res
