""" 
losses.py

In this file I make sure that the dice loss is used because for segmentation the majority class pixel 
is background=0 and foreground=1, so the model doesn't skip the foreground class. I am using dice loss
along with binary cross entropy loss to make it much stronger.
"""

import torch 
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        # binary cross entropy loss
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # calculating the binary cross entropy loss
        bce_loss = self.bce(outputs, targets)

        # converting the raw logits from outputs into probability
        outputs_sigmoid = torch.sigmoid(outputs)
        # flatten the output and target values so dice is computed over all pixels
        outputs_flat = outputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        # dice formula = (2 * intersection + smooth) / (sum of both + smooth)
        intersection = (outputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (outputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        return bce_loss + dice_loss


def calculate_dice_score(outputs, targets, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    # convert boolean mask (True/False) to float (1.0/0.0) so arithmetic works
    outputs = (outputs > threshold).float()
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    # adding 1e-8 to prevent division by zero in case both output and target are all zeros
    dice = (2. * intersection) / (outputs.sum() + targets.sum() + 1e-8)

    return dice.item()  # .item() converts single-element tensor → Python float
