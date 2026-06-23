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
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # newly added import
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras_flops import get_flops

from util import get_max_shape, round_up_to_multiple, pad_training_images as pad_images
from sa_unet import SA_UNetV2
from loss import combined_loss


weight = "STARE/Model/SA_UNetv2.h5"
restore = False


def load_stare_data(image_dir, label_dir, target_shape):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    x_data, y_data = [], []

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.ah.png"

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            print(f"⚠️ Missing label: {label_path}, skipping")
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('L'))

        img_padded = pad_images(img, target_shape)
        label_padded = pad_images(label, target_shape)

        _, label_bin = cv2.threshold(label_padded, 127, 255, cv2.THRESH_BINARY)
        label_bin = np.expand_dims(label_bin, axis=-1)

        x_data.append(img_padded)
        y_data.append(label_bin)

    x_data = np.array(x_data, dtype=np.float32) / 255.0
    y_data = np.array(y_data, dtype=np.float32) / 255.0
    return x_data, y_data


if __name__ == "__main__":
    data_location = ""
    train_image_dir = data_location + 'STARE/train/images/'
    train_label_dir = data_location + 'STARE/train/labels/'

    # Step 1: Get max shape from training set
    train_max_h, train_max_w = get_max_shape(train_image_dir)
    # target_h = round_up_to_multiple(train_max_h, 8)
    # target_w = round_up_to_multiple(train_max_w, 8)
    # target_shape = (target_h, target_w)
    desired_size = round_up_to_multiple(max(train_max_h, train_max_w), 8)
    target_shape = (desired_size, desired_size)

    print(f"Max train image size: ({train_max_h}, {train_max_w}) -> Padded to: {target_shape}")

    # Step 2: Load and pad images
    x_all, y_all = load_stare_data(train_image_dir, train_label_dir, target_shape)

    # Step 3: Split train/val (90/10)
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.1, shuffle=True, random_state=42
    )

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape)

    # desired_size = 704
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
    # Calculate FLOPs (using 1 sample as input, shape = 592x592x3)
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
            patience=20,      # wait for 20 epochs without improvement
            min_lr=1e-8,      # minimum learning rate
            verbose=1
        ),
        
        # Early stopping mechanism (monitoring validation loss)
        EarlyStopping(
            monitor='val_loss',
            patience=30,      # stop after 30 epochs without improvement
            restore_best_weights=True,  # restore the best weights
            verbose=1
        )
    ]
   
    # Start training
    history = model.fit(
        x_train, y_train,
        epochs=150,
        batch_size=2,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=callbacks  # use the configured callback list
    )    
