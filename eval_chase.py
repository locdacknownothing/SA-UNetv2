import os

import numpy as np 
import cv2
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, jaccard_score

from util import pad_images, restore_images
from sa_unet import SA_UNetV2


weight = "CHASE/Model/SA_UNetv2.h5"
save_dir = 'CHASE/SA_UNetv2/save_dir'  # Path of the folder where you want to save files
os.makedirs(save_dir, exist_ok=True)  # Automatically create the directory if it does not exist

# groundtruth_dir = 'CHASE/SA_UNetv2/groundtruth_dir'  # Path of the folder for saving ground truth
# os.makedirs(groundtruth_dir, exist_ok=True)  # Automatically create the directory if it does not exist

# with_mask = False


def get_data_label_from_files(test_files, testing_images_loc, testing_label_loc):
    test_data = []
    test_label = []

    # sample_data_names = ["Image_11L.jpg"]
    # sample_label_names = ["Image_11L_1stHO.png"]
    
    for i in test_files:
        im = imageio.imread(testing_images_loc + i)
        label = imageio.imread(testing_label_loc + i.split('.')[0] + '_1stHO.png', mode="L")
        assert im.shape[:2] == (960, 999), im.shape
        assert label.shape[:2] == (960, 999), label.shape
        
        test_data.append(im)
        _, temp = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        test_label.append(temp)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_label = np.expand_dims(test_label,axis=-1)

    return test_data, test_label


# def get_data_label_mask_from_files(test_files, testing_images_loc, testing_label_loc, mask_loc):
#     # Read test images and labels
#     test_data = []
#     test_label = []
#     mask_label = []
#     for i in test_files:
#         im = imageio.imread(testing_images_loc + i)
#         label = imageio.imread(testing_label_loc + i.split('_')[0] + '_manual1.gif')
#         mask = imageio.imread(mask_loc + i.split('.')[0] + '_mask.gif')
#         mask_label.append(mask)

#         test_data.append(cv2.resize(im, (960, 999)))
#         temp = cv2.resize(label, (960, 999))
#         _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
#         test_label.append(temp)    

#     test_data = np.array(test_data)
#     test_label = np.array(test_label)
#     mask_label = np.array(mask_label)

#     return test_data, test_label, mask_label


if __name__ == "__main__":
    data_location = ""
    testing_images_loc = data_location + 'CHASE/test/image/'
    testing_label_loc = data_location + 'CHASE/test/label/'
    test_files = os.listdir(testing_images_loc)

    test_data, test_label = get_data_label_from_files(
        test_files,
        testing_images_loc,
        testing_label_loc,
    )

    x_test = test_data.astype('float32') / 255.
    y_test = test_label.astype('float32') / 255.

    print('x_test shape:', x_test.shape)
    print('y_test shape:', y_test.shape)

    desired_size = 1008
    model = SA_UNetV2(
        input_size=(desired_size, desired_size, 3),
        block_size=7,
        rate=0.15,
        start_neurons=16,
    )
    if os.path.isfile(weight):
        model.load_weights(weight)

    x_test_padded = pad_images(x_test,(len(x_test), desired_size, desired_size, 3))
    y_pred = model.predict(x_test_padded, batch_size=1)
    y_pred= restore_images(y_pred, y_test.shape)

    # Initialize lists
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    auc_scores = []
    mcc_scores = []
    f1_scores = []
    jaccard_scores = []

    # Iterate through predictions and ground truth labels
    for i, (y_pred_image, y_test_image) in enumerate(zip(y_pred, y_test)):
        # ➤ Binarize prediction image
        _, binary_pred = cv2.threshold(y_pred_image, 0.5, 1, cv2.THRESH_BINARY)
        
        # ➤ Flatten to 1D arrays
        y_pred_flat = binary_pred.ravel()
        y_test_flat = y_test_image.ravel()

        # ➤ Compute metrics
        tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_flat).ravel()
        accuracy = accuracy_score(y_test_flat, y_pred_flat)
        sensitivity = recall_score(y_test_flat, y_pred_flat)
        specificity = tn / (tn + fp + 1e-6)  # Avoid division by zero
        auc = roc_auc_score(y_test_flat, y_pred_image.ravel())
        mcc = matthews_corrcoef(y_test_flat, y_pred_flat)
        f1 = f1_score(y_test_flat, y_pred_flat)
        jaccard = jaccard_score(y_test_flat, y_pred_flat)

        # ➤ Save metrics
        accuracy_scores.append(accuracy)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
        auc_scores.append(auc)
        mcc_scores.append(mcc)
        f1_scores.append(f1)
        jaccard_scores.append(jaccard)

        # ➤ Save prediction and ground truth images (binary images: 0 or 255)
        pred_save = (binary_pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{i}_bin.png'), pred_save)

        # gt_save = (y_test_image * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(groundtruth_dir, f'{i}.png'), gt_save)

        # Save probability map (0~255)
        prob_img = (y_pred_image * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{i}_prob.png'), prob_img)

    # ➤ Print average metrics
    print('Average Accuracy:', np.mean(accuracy_scores))
    print('Average Sensitivity:', np.mean(sensitivity_scores))
    print('Average Specificity:', np.mean(specificity_scores))
    print('Average AUC:', np.mean(auc_scores))
    print('Average MCC:', np.mean(mcc_scores))
    print('Average F1 Score:', np.mean(f1_scores))
    print('Average Jaccard Index:', np.mean(jaccard_scores))
