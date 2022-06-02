import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
import random

sm.set_framework('tf.keras')
sm.framework()

from keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow.keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from sklearn import metrics

import math
import pandas as pd

from tensorflow.keras.layers import Input

# import seaborn as sns

# import ast

# from osgeo import gdal
import numpy as np
import os
# import subprocess
# import glob

def preprocess_data(img, mask, num_class):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    mask = to_categorical(mask, num_class)
      
    return (img,mask)

def trainGenerator(train_img_path, train_mask_path, num_class):
    
    img_data_gen_args = dict(horizontal_flip = True,
                      vertical_flip = True,
                      fill_mode = 'reflect')
    
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        target_size = (4096, 4096),
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        target_size = (4096, 4096),
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
    
    train_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)

if __name__ == '__main__':

    seed = 24
    batch_size = 3
    n_classes = 8

    test_img_path = '/workspace/project_4413/kaggle_data/main_test_image/'
    test_msk_path = '/workspace/project_4413/kaggle_data/main_test_mask/'

    test_img_gen = trainGenerator(test_img_path, test_msk_path, num_class = n_classes)

    x, y = test_img_gen.__next__()

    model_names = ['Unet-resnet50', 'Unet-resnet101', 'Unet-resnet152']
    models_path = '/workspace/project_4413/model_stats/'

    models = []
    for m in model_names:
        md = m.split('-')[0]
        bk = m.split('-')[1]

        models.append(load_model(models_path + f'{md}_{bk}/model.hdf5', compile = False))
    
    model = models[0]

    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    backbone = 'resnet50'
    lr = 0.001
    new_model = sm.Unet(backbone_name = backbone, 
                            encoder_weights = 'imagenet', 
                            input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                            # inputs = [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS],
                            classes = n_classes, 
                            activation = 'softmax')
    
    new_model.compile(optimizers.Adam(learning_rate = lr), 
                          loss = sm.losses.categorical_focal_jaccard_loss, 
                          metrics = [sm.metrics.iou_score])
    
    new_model.set_weights(model.get_weights())

    test_image_batch, test_mask_batch = test_img_gen.__next__()

    test_mask_batch_argmax = np.argmax(test_mask_batch, axis = 3) 
    test_pred_batch = new_model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis = 3)

    IOU_keras = MeanIoU(num_classes = n_classes)
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax) 
    print("Mean IoU =", IOU_keras.result().numpy())

    pixel_acc = []
    test_image_batch, test_mask_batch = test_img_gen.__next__()

    test_mask_batch_argmax = np.argmax(test_mask_batch, axis = 3) 
    test_pred_batch = new_model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis = 3)

    for j in range(batch_size):
        correct = test_pred_batch_argmax[j] == test_mask_batch_argmax[j]
        
        


        
    
    print(f'Max Pixel Accuracy = {np.max(pixel_acc)}, Mean Pixel Accuracy = {np.mean(pixel_acc)}')
