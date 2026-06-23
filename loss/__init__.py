from tensorflow import keras

from .base import (
    dice_loss, 
    jaccard_loss, 
    ssim_loss, 
    gradient_regularizer,
    MCC_Loss,
)
from .cldice_2d import soft_clDice_loss


cl_dice_loss = soft_clDice_loss(iter_=15)


def combined_loss(y_true, y_pred, alpha=0.5):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    jaccard = jaccard_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    smoothness = gradient_regularizer(y_pred)
    mcc = MCC_Loss()(y_true, y_pred) 
    
    cl_dice = cl_dice_loss(y_true, y_pred)
    
    return alpha * bce + (1 - alpha) * mcc
    # return alpha * dice + (1 - alpha) * cl_dice
