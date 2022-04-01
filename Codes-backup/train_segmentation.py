import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm

import json

# import segmentation_models_pytorch as sm
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

import wandb
from wandb.keras import WandbCallback

wandb.login()

sm.set_framework('tf.keras')
sm.framework()


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


parser = argparse.ArgumentParser()
parser.add_argument('--tr_img', help = 'path for training images')
parser.add_argument('--tr_msk', help = 'path for training masks')

parser.add_argument('--v_img', help = 'path for validation images')
parser.add_argument('--v_msk', help = 'path for validation masks')

parser.add_argument('--ts_img', help = 'path for testing images')
parser.add_argument('--ts_msk', help = 'path for testing masks')

parser.add_argument('--m_dir', help = 'statistics directory')

# parser.add_argument('--backbone', help = 'segmentation model backbone')
# parser.add_argument('--model', help = 'segmentation model')
parser.add_argument('--bs', help = 'batch size')

parser.add_argument('--opt', help = 'optimizer')
parser.add_argument('--lr', help = 'learning rate')

parser.add_argument('--eps', help = 'no. of epochs')
parser.add_argument('--res', help = 'image resolution')

args = parser.parse_args()

models = ['Unet-resnet101', 'Unet-resnet152', 'Unet-densenet121', 'Unet-densenet169', 'Unet-vgg19',
               'FPN-resnext50', 'FPN-densenet121', 'Linknet-inceptionv3', 'FPN-vgg16', 'Linknet-mobilenetv2']

# models = ['Unet-resnet101', 'Unet-resnet152']

# def img_display(x, y):
#     for i in range(0,1):
#         image = x[i]

#         # print(image)
#         mask = np.argmax(y[i], axis=2)
#         plt.subplot(1,2,1)
#         plt.imshow(image)
#         plt.subplot(1,2,2)
#         plt.imshow(mask, cmap='gray')
#         # plt.show()

def loss_iou(history, dir):
    plt.clf()

    plt.figure(figsize = (10, 10), dpi = 125)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dir}/loss.jpg', bbox_inches = 'tight')
    # plt.show()

    plt.clf()

    plt.figure(figsize = (10, 10), dpi = 125)
    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(f'{dir}/iou.jpg', bbox_inches = 'tight')
    # plt.show()

def train_loss_cmp(history, dir):
    i = 0
    plt.figure(figsize = (10, 10), dpi = 125)
    for model in models:
        plt.plot(history[i].history['loss'], marker='o')
        i = i + 1
    plt.title('Score per epoch'); plt.ylabel('Train Loss')
    plt.xlabel('epoch')
    plt.legend(labels = models)
    plt.savefig(f'{dir}/Train_Loss_Comparison.jpg', bbox_inches = 'tight')
    # plt.show()

def main():

    run = wandb.init(project='my-t-4',
                 config={  # and include hyperparameters and metadata
                     "learning_rate": 0.001,
                     "epochs": 2,
                     "batch_size": 16,
                 })
    config = wandb.config 

    seed = 24
    batch_size = int(args.bs)
    resolution = int(args.res)
    n_classes = 5


    # Loading Training Data
    train_img_path = args.tr_img
    train_mask_path = args.tr_msk
    train_img_gen = trainGenerator(train_img_path, train_mask_path, n_classes, resolution, batch_size, seed)

    # Loading Validation Data
    val_img_path = args.v_img
    val_mask_path = args.v_msk
    val_img_gen = trainGenerator(val_img_path, val_mask_path, n_classes, resolution, batch_size, seed)

    x, y = train_img_gen.__next__()
    x_val, y_val = val_img_gen.__next__()

    num_train_imgs = len(os.listdir(args.tr_img + 'train/'))
    num_val_images = len(os.listdir(args.v_img + 'val/'))
    steps_per_epoch = num_train_imgs//batch_size
    val_steps_per_epoch = num_val_images//batch_size

    print(num_train_imgs, num_val_images, steps_per_epoch, val_steps_per_epoch)

    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]

    lr = float(args.lr)
    eps = int(args.eps)

    main_history = []
    # history_dict = {}

    for m in models:
        model_name = m.split('-')[0] 
        backbone = m.split('-')[1]

        model = None
        if model_name == 'Unet':
            model = sm.Unet(backbone_name = backbone, 
                            encoder_weights = 'imagenet', 
                            input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                            classes = n_classes, 
                            activation = 'softmax')
        
        
        if model_name == 'FPN':
            model = sm.FPN(backbone_name = backbone, 
                           encoder_weights = 'imagenet', 
                           input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                           classes = n_classes, 
                           activation = 'softmax')
        
        
        if model_name == 'PSPNet':
            model = sm.PSPNet(backbone_name = backbone, 
                              encoder_weights = 'imagenet', 
                              input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                              classes = n_classes, 
                              activation = 'softmax')
        
        if model_name == 'Linknet':
            model = sm.Linknet(backbone_name = backbone, 
                               encoder_weights = 'imagenet', 
                               input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                               classes = n_classes, 
                               activation = 'softmax')


        if args.opt == 'Adam':
            model.compile(optimizers.Adam(learning_rate = lr), 
                          loss = sm.losses.categorical_focal_jaccard_loss, 
                          metrics = [sm.metrics.iou_score])
        
        history = model.fit(train_img_gen,
                            steps_per_epoch = steps_per_epoch,
                            epochs = eps,
                            verbose = 1,
                            validation_data = val_img_gen,
                            validation_steps = val_steps_per_epoch,
                            callbacks = [WandbCallback()])
        
        history_dict = {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'iou':history.history['iou_score'],
            'val_iou': history.history['val_iou_score']
        }
        
        main_history.append(history)

        model_name = f'{model_name}_{backbone}'
        m_dir = args.m_dir

        os.mkdir(f'{m_dir}{model_name}')
        model.save(f'{m_dir}/{model_name}/model.hdf5')

        dir = f'{m_dir}/{model_name}/'
        loss_iou(history, dir)

        with open(f'{m_dir}/{model_name}/history_dict_{model_name}.txt', 'w') as f:
            f.write(json.dumps(history_dict))

    # print(main_history[0].history['loss'])
    train_loss_cmp(main_history, m_dir)
    

    run.finish()


# def main():
#     seed = 24
#     batch_size = int(args.bs)
#     resolution = int(args.res)
#     n_classes = 5

#     BACKBONE = args.backbone

#     train_img_path = args.tr_img
#     train_mask_path = args.tr_msk

#     train_img_gen = trainGenerator(train_img_path, train_mask_path, n_classes, resolution, batch_size, seed)

#     val_img_path = args.v_img
#     val_mask_path = args.v_msk
#     val_img_gen = trainGenerator(val_img_path, val_mask_path, n_classes, resolution, batch_size, seed)

#     x, y = train_img_gen.__next__()
#     x_val, y_val = val_img_gen.__next__()

#     # img_display(x, y)

#     num_train_imgs = len(os.listdir(args.tr_img + 'train/'))
#     num_val_images = len(os.listdir(args.v_img + 'val/'))
#     steps_per_epoch = num_train_imgs//batch_size
#     val_steps_per_epoch = num_val_images//batch_size

#     print(num_train_imgs, num_val_images, steps_per_epoch, val_steps_per_epoch)

#     IMG_HEIGHT = x.shape[1]
#     IMG_WIDTH  = x.shape[2]
#     IMG_CHANNELS = x.shape[3]

#     model = None
#     if args.model == 'Unet':
#         model = sm.Unet(backbone_name = BACKBONE, 
#                         encoder_weights = 'imagenet', 
#                         input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
#                         classes = n_classes, 
#                         activation = 'softmax')

#     lr = float(args.lr)
#     eps = int(args.eps)

#     if args.opt == 'Adam':
#         model.compile(optimizers.Adam(learning_rate = lr), 
#                       loss = sm.losses.categorical_focal_jaccard_loss, 
#                       metrics = [sm.metrics.iou_score])
    
#     history = model.fit(train_img_gen,
#           steps_per_epoch = steps_per_epoch,
#           epochs = eps,
#           verbose = 1,
#           validation_data = val_img_gen,
#           validation_steps = val_steps_per_epoch)

#     model_name = f'{args.model}_{BACKBONE}'
#     m_dir = args.m_dir

#     os.mkdir(f'{m_dir}/{model_name}')
#     model.save(f'{m_dir}/{model_name}/model.hdf5')

#     dir = f'{m_dir}/{model_name}/'
#     loss_iou(history, dir)



if __name__ == '__main__':
    main()

    