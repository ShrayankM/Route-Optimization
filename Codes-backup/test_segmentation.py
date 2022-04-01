import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

import keras
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.utils import to_categorical

import argparse

sm.set_framework('tf.keras')
sm.framework()

from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--ts_img', help = 'path for testing images')
parser.add_argument('--ts_msk', help = 'path for testing masks')
parser.add_argument('--m_path', help = 'path for trained model')

parser.add_argument('--bs', help = 'batch size')
parser.add_argument('--res', help = 'image resolution')

args = parser.parse_args()

def preprocess_data(img, mask, num_class):
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    mask = to_categorical(mask, num_class)
      
    return (img,mask)

def trainGenerator(train_img_path, train_mask_path, num_class, img_res, batch_size, seed):

    img_data_gen_args = dict(horizontal_flip = True,
                      vertical_flip = True,
                      fill_mode = 'reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        target_size = (img_res, img_res),
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size = (img_res, img_res),
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)

def check_ious(test_img_gen, itrs, model):
    ious = []
    for i in range(itrs):
        test_image_batch, test_mask_batch = test_img_gen.__next__()
        
        img_num = random.randint(0, test_image_batch.shape[0]-1)

        #Convert categorical to integer for visualization and IoU calculation
        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

        n_classes = 5
        IOU_keras = MeanIoU(num_classes = n_classes)  
        IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
        # print("Mean IoU =", IOU_keras.result().numpy())

        ious.append(IOU_keras.result().numpy())
    return ious


def main():
    model = load_model(f'{args.m_path}/model.hdf5', compile = False)
    n_classes = 5

    batch_size = int(args.bs)
    seed = 24
    img_res = int(args.res)

    test_img_path = args.ts_img
    test_mask_path = args.ts_msk
    test_img_gen = trainGenerator(test_img_path, test_mask_path, n_classes, img_res, batch_size, seed)

    num_test_imgs = len(os.listdir(test_img_path + '/test'))
    itrs = num_test_imgs // batch_size

    ious = check_ious(test_img_gen, itrs, model)
    print(f'Max IoU = {np.max(ious)}, Average IoU = {np.mean(ious)}')

if __name__ == '__main__':
    main()