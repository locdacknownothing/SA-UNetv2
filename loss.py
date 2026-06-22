import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K


def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    loss = 1 - (2. * intersection + smooth) / (union + smooth)
    return loss

def jaccard_loss(y_true, y_pred, smooth=1e-6):
    """Jaccard Loss (Intersection over Union) for binary segmentation."""
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    loss = 1 - (intersection + smooth) / (union + smooth)
    return loss

def ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1 - tf.reduce_mean(ssim)

def gradient_regularizer(y_pred):
    dx = tf.abs(y_pred[:,1:,:,:] - y_pred[:,:-1,:,:])
    dy = tf.abs(y_pred[:,:,1:,:] - y_pred[:,:,:-1,:])
    return tf.reduce_mean(dx) + tf.reduce_mean(dy)


class MCC_Loss(keras.losses.Loss):
    def __init__(self, name="mcc_loss"):
        super(MCC_Loss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        """
        Calculates the Matthews Correlation Coefficient (MCC) loss.
        
        Arguments:
            y_true: Ground truth labels, shape (batch_size, num_classes).
            y_pred: Model predictions (probabilities), shape (batch_size, num_classes).
            
        Returns:
            A scalar value representing the MCC loss.
        """
        # Keep the predictions as continuous probabilities instead of applying a hard threshold
        tp = K.sum(y_pred * y_true)  # True Positives
        tn = K.sum((1 - y_true) * (1 - y_pred))  # True Negatives
        fp = K.sum((1 - y_true) * y_pred)  # False Positives
        fn = K.sum(y_true * (1 - y_pred))  # False Negatives
        
        # Compute MCC using the formula
        numerator = tp * tn - fp * fn
        denominator = K.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        
        # MCC loss: 1 - MCC
        mcc = numerator / (denominator + K.epsilon())  # Adding epsilon to avoid divide by zero
        return 1 - mcc  # We return 1 - MCC because we are minimizing the loss


def combined_loss(y_true, y_pred, alpha=0.5):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    jaccard = jaccard_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    smoothness = gradient_regularizer(y_pred)
    mcc = MCC_Loss()(y_true, y_pred) 
    
    return alpha * bce + (1 - alpha) * mcc
