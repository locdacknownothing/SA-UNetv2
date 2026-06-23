import os

import numpy as np 
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, jaccard_score

# from util import pad_images, restore_images
from sa_unet import SA_UNetV2


weight = "STARE/Model/SA_UNetv2.h5"
save_dir = 'STARE/SA_UNetv2/save_dir'  # Path of the folder where you want to save files
os.makedirs(save_dir, exist_ok=True)  # Automatically create the directory if it does not exist

# groundtruth_dir = 'STARE/SA_UNetv2/groundtruth_dir'  # Path of the folder for saving ground truth
# os.makedirs(groundtruth_dir, exist_ok=True)  # Automatically create the directory if it does not exist


def get_data_label_from_files(test_files, testing_images_loc, testing_label_loc):
    test_data = []
    test_label = []
    
    target_size = None  # Will store the size of the first image (W, H)
    
    for fname in test_files:
        # Construct label filename using .ah.png convention
        label_name = fname.replace('.png', '.ah.png')
        img_path = os.path.join(testing_images_loc, fname)
        label_path = os.path.join(testing_label_loc, label_name)

        if not os.path.exists(label_path):
            print(f"Missing label file: {label_path}, skipping")
            continue

        # Load image and label
        im = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('L'))

        # Save first image size
        if target_size is None:
            target_size = (im.shape[1], im.shape[0])  # (W, H)

        # Resize if needed
        if (im.shape[1], im.shape[0]) != target_size:
            im = cv2.resize(im, target_size)
        if (label.shape[1], label.shape[0]) != target_size:
            label = cv2.resize(label, target_size)

        # Binarize label
        _, label_bin = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        label_bin = np.expand_dims(label_bin, axis=-1)

        test_data.append(im)
        test_label.append(label_bin)
    
    return test_data, test_label, target_size


if __name__ == "__main__":
    data_location = ""
    testing_images_loc = data_location + 'STARE/test/images/'
    testing_label_loc = data_location + 'STARE/test/labels/'
    # Only read .png images
    test_files = sorted([f for f in os.listdir(testing_images_loc) if f.endswith('.png')])

    test_data, test_label, target_size = get_data_label_from_files(
        test_files,
        testing_images_loc,
        testing_label_loc,
    )

    x_test = np.array(test_data, dtype=np.float32) / 255.
    y_test = np.array(test_label, dtype=np.float32) / 255.

    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)


    desired_size = 704

    # List to store original shapes
    original_shapes = []

    # List to store padded images
    padded_test_data = []

    # Automatically get original sizes and pad images
    for i in range(x_test.shape[0]):
        img = x_test[i]
        h, w = img.shape[:2]
        original_shapes.append((h, w))
        
        # Create a blank image of the target size
        padded_img = np.zeros((desired_size, desired_size, 3), dtype=np.float32)
        padded_img[:h, :w, :] = img  # Place original image
        padded_test_data.append(padded_img)

    x_test_padded = np.array(padded_test_data)
    print("padded x_test shape:", x_test_padded.shape)

    model = SA_UNetV2(
        input_size=(desired_size, desired_size, 3),
        block_size=7,
        rate=0.15,
        start_neurons=16,
    )
    if os.path.isfile(weight):
        model.load_weights(weight)

    y_pred_padded = model.predict(x_test_padded)
    print("y_pred padded shape:", y_pred_padded.shape)

    # Restore predictions to original sizes
    y_pred_restored = []

    for i in range(y_pred_padded.shape[0]):
        h, w = original_shapes[i]
        restored = y_pred_padded[i, :h, :w, :]
        y_pred_restored.append(restored)
    
    y_pred = np.array(y_pred_restored)
    print("y_pred restored shape:", y_pred.shape)


    # Initialize lists
    # y_pred_threshold = []
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    auc_scores = []
    mcc_scores = []
    f1_scores = []
    jaccard_scores = []

    # Iterate through predictions and ground truth labels
    for i, (y_pred_image, y_test_image) in enumerate(zip(y_pred, y_test)):
        # Binarize predictions
        _, temp = cv2.threshold(y_pred_image, 0.5, 1, cv2.THRESH_BINARY)
        # y_pred_threshold.append(temp)

        # Flatten arrays for metric calculations
        y_pred_flat = np.ravel(temp)
        y_test_flat = np.ravel(y_test_image)

        # Compute confusion matrix and metrics
        tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat).ravel()
        accuracy = accuracy_score(y_test_flat, y_pred_flat)
        sensitivity = recall_score(y_test_flat, y_pred_flat)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_test_flat, np.ravel(y_pred_image))
        mcc = matthews_corrcoef(y_test_flat, y_pred_flat)
        f1 = f1_score(y_test_flat, y_pred_flat)
        jaccard = jaccard_score(y_test_flat, y_pred_flat)

        # Append metrics to lists
        accuracy_scores.append(accuracy)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        auc_scores.append(auc)
        mcc_scores.append(mcc)
        f1_scores.append(f1)
        jaccard_scores.append(jaccard)

        # Save probability map (0~255)
        prob_img = (y_pred_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{i}_prob.png'), prob_img)

        # Save binary image (0 or 255)
        bin_img = (temp * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{i}_bin.png'), bin_img)

    # ➤ Print average metrics
    print('Average Accuracy:', np.mean(accuracy_scores))
    print('Average Sensitivity:', np.mean(sensitivity_scores))
    print('Average Specificity:', np.mean(specificity_scores))
    print('Average AUC:', np.mean(auc_scores))
    print('Average MCC:', np.mean(mcc_scores))
    print('Average F1 Score:', np.mean(f1_scores))
    print('Average Jaccard Index:', np.mean(jaccard_scores))
