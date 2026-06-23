import random
import threading, os, time
import logging
import time

import cv2
from PIL import Image, ImageEnhance, ImageFile
import numpy as np


logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

opsList = {
    "randomRotation", 
    "randomColor", 
    "randomGaussian",
}

singleOpsList =  {
    "horizontalFlip",
    "verticalFlip",
    "diagonalFlip",
}


class DataAugmentation:
    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):

        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    # @staticmethod
    # def randomCrop(image, label):

    #     image_width = image.size[0]
    #     image_height = image.size[1]
    #     crop_win_size = np.random.randint(40, 68)
    #     random_region = (
    #         (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
    #         (image_height + crop_win_size) >> 1)
    #     return image.crop(random_region), label

    @staticmethod
    def randomColor(image, label):

        random_factor = np.random.randint(0, 31) / 10.  
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
        random_factor = np.random.randint(0, 31) / 10.  
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):

        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.array(image)
        # img.flags.writeable = 1
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label

    @staticmethod
    def horizontalFlip(image, label):
        img = np.asarray(image)
        h_flip_img = cv2.flip(img, 1)
        lbl = np.asarray(label)
        h_flip_lbl = cv2.flip(lbl, 1)
        return Image.fromarray(h_flip_img), Image.fromarray(h_flip_lbl)

    @staticmethod
    def verticalFlip(image, label):
        img = np.asarray(image)
        v_flip_img = cv2.flip(img, 0)
        lbl = np.asarray(label)
        v_flip_lbl = cv2.flip(lbl, 0)
        return Image.fromarray(v_flip_img), Image.fromarray(v_flip_lbl)

    @staticmethod
    def diagonalFlip(image, label):
        img = np.asarray(image)
        hv_flip_img = cv2.flip(img, -1)
        lbl = np.asarray(label)
        hv_flip_lbl = cv2.flip(lbl, -1)
        return Image.fromarray(hv_flip_img), Image.fromarray(hv_flip_lbl)

    @staticmethod
    def saveImage(image, path, suffix=".png"):
        # new_path = ".".join(str(path).split(".")[:-1]) + suffix
        image.save(path)


def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception:
        print(str(Exception))


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=3):
    funcMap = {
        "randomRotation": DataAugmentation.randomRotation,
        # "randomCrop": DataAugmentation.randomCrop,
        "randomColor": DataAugmentation.randomColor,
        "randomGaussian": DataAugmentation.randomGaussian,
    }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
    
    makeDir(img_des_path)
    makeDir(label_des_path)

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))

def imageSingleOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name):
    funcMap = {
        "horizontalFlip": DataAugmentation.horizontalFlip,
        "verticalFlip": DataAugmentation.verticalFlip,
        "diagonalFlip": DataAugmentation.diagonalFlip,
    }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
    
    makeDir(img_des_path)
    makeDir(label_des_path)

    new_image, new_label = funcMap[func_name](image, label)
    DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + "0" + img_file_name))
    DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + "0" + label_file_name))

def threadOPS(img_path, new_img_path, label_path, new_label_path):

    # img path
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
    else:
        label_names = [label_path]

    img_num = 0
    label_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1

    # label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1

    if img_num != label_num:
        print('the num of img and label is not equal')
        exit()
    else:
        num = img_num

    for i in range(num):
        img_name = img_names[i]
        label_name = label_names[i]

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        # print(tmp_img_name)
        image = DataAugmentation.openImage(tmp_img_name)
        label = DataAugmentation.openImage(tmp_label_name)

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(
                target=imageOps,
                args=(ops_name, image, label, new_img_path, new_label_path, img_name, label_name),
            )
            threadImage[_index].start()
            _index += 1
            time.sleep(5)

        threadImage = [0] * 3
        _index = 0
        for single_ops_name in singleOpsList:
            threadImage[_index] = threading.Thread(
                target=imageSingleOps,
                args=(single_ops_name, image, label, new_img_path, new_label_path, img_name, label_name),
            )
            threadImage[_index].start()
            _index += 1
            time.sleep(5)


if __name__ == "__main__":
    dirs = [
        "DRIVE/train/images",  # set your path of training images
        "DRIVE/train_aug/images",
        "DRIVE/train/labels",  # set your path of training labels
        "DRIVE/train_aug/labels",
    ]
    
    # Please modify the path
    threadOPS(*dirs)
