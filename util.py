import os

import numpy as np
import cv2
from PIL import Image


def pad_images(input_data, target_shape):
    # Create an array filled with zeros for padding
    padded_data = np.zeros(target_shape)
    for i in range(input_data.shape[0]):
        # Copy original image into the padded array
        padded_data[i, :input_data.shape[1], :input_data.shape[2], :] = input_data[i]
    return padded_data

def restore_images(output_data, original_shape):
    # Create an array to store restored images
    restored_data = np.zeros(original_shape)
    for i in range(original_shape[0]):
        # Crop the padded image back to original size
        restored_data[i] = output_data[i, :original_shape[1], :original_shape[2], :]
    return restored_data

def convert_array(data, desired_size, channels):
    x = data.astype('float32') / 255.
    # adapt this if using `channels_first` image data format
    x = np.reshape(x, (len(x), desired_size, desired_size, channels))
    return x  

def get_max_shape(image_dir):
    max_h, max_w = 0, 0
    for file in sorted(os.listdir(image_dir)):
        if file.endswith('.png'):
            img = Image.open(os.path.join(image_dir, file))
            h, w = img.size[1], img.size[0]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
    return max_h, max_w

def round_up_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple

def pad_training_images(img, target_shape):
    h, w = img.shape[:2]
    target_h, target_w = target_shape

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 3:
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    else:
        return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
