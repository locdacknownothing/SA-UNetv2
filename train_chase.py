import tensorflow as tf

# Configure TensorFlow session for dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import os
from pathlib import Path

import numpy as np 
import cv2
import imageio
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # newly added import
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras_flops import get_flops

from util import pad_images, convert_array
from sa_unet import SA_UNetV2
from loss import combined_loss


weight = "CHASE/Model/SA_UNetv2.h5"
restore = False


def get_data_label_from_files(files, images_loc, label_loc, desired_size):
    data = []
    labels = []

    # sample_data_names = ["hImage_01L.jpg", "randomGaussian0Image_04R.jpg"]
    # sample_label_names = ["hImage_01L_1stHO.png", "randomGaussian0Image_04R_1stHO.png"]

    for i in files:
        im = imageio.imread(images_loc + i)
        # print(im.shape)
        im_reshaped = np.reshape(im, (1, *im.shape[:2], 3))
        new_im = pad_images(im_reshaped, (1,desired_size,desired_size,3))
        new_im = np.reshape(new_im, (desired_size, desired_size, 3))
        label = imageio.imread(label_loc + i.split('.')[0] + '_1stHO.png', mode='L')
        # print(label.shape)
        label_reshaped = np.reshape(label, (1, *im.shape[:2], 1))
        new_label = pad_images(label_reshaped, (1,desired_size,desired_size,1))
        new_label =  np.reshape(new_label, (desired_size, desired_size, 1))
        data.append(cv2.resize(new_im, (desired_size, desired_size)))
        temp = cv2.resize(new_label, (desired_size, desired_size))
        _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        labels.append(temp)

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    data_location = ""
    training_images_loc = data_location + 'CHASE/train/image/'
    training_label_loc = data_location + 'CHASE/train/label/'

    validate_images_loc = data_location + 'CHASE/validate/images/'
    validate_label_loc = data_location + 'CHASE/validate/labels/'
    train_files = os.listdir(training_images_loc)
    validate_files = os.listdir(validate_images_loc)
    desired_size=1008

    train_data, train_label = get_data_label_from_files(
        train_files, 
        training_images_loc,
        training_label_loc,
        desired_size,
    )
    validate_data, validate_label = get_data_label_from_files(
        validate_files, 
        validate_images_loc,
        validate_label_loc,
        desired_size,
    )

    x_train = convert_array(train_data, desired_size, 3)
    y_train = convert_array(train_label, desired_size, 1)

    x_validate = convert_array(validate_data, desired_size, 3)
    y_validate = convert_array(validate_label, desired_size, 1)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_validate shape:', x_validate.shape)
    print('y_validate shape:', y_validate.shape)

    model = SA_UNetV2(
        input_size=(desired_size, desired_size, 3),
        block_size=7,
        rate=0.15,
        start_neurons=16,
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=['accuracy']
    )

    model.summary()
    # Calculate FLOPs (using 1 sample as input, shape = desired_size x desired_size x 3)
    flops = get_flops(model, batch_size=1)
    print(f"GFLOPs: {flops / 1e9:.3f} GFLOPs")

    if restore and os.path.isfile(weight):
        model.load_weights(weight)

    weight_dir = Path(weight).parent
    weight_dir.mkdir(exist_ok=True, parents=True)

    # Configure callbacks
    callbacks = [
        # Save best weights (monitoring validation accuracy)
        ModelCheckpoint(
            weight,
            monitor='val_accuracy',  # you can also use 'val_loss'
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='max'  # use max when monitoring accuracy, use min when monitoring loss
        ),
        
        # Dynamic learning rate reduction (monitoring validation loss)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,       # learning rate reduction factor
            patience=10,      # wait for 10 epochs without improvement
            min_lr=1e-6,      # minimum learning rate
            verbose=1
        ),
        
        # Early stopping mechanism (monitoring validation loss)
        EarlyStopping(
            monitor='val_loss',
            patience=20,      # stop after 20 epochs without improvement
            restore_best_weights=True,  # restore the best weights
            verbose=1
        )
    ]
   
    # Start training
    history = model.fit(
        x_train, y_train,
        epochs=150,
        batch_size=4,
        validation_data=(x_validate, y_validate),
        shuffle=True,
        callbacks=callbacks  # use the configured callback list
    )    
