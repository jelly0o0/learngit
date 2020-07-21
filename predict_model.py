# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
import math
import random

BATCH_SIZE = 32
IMG_WIDTH = 128
IMG_HEIGHT = 128


def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, ratio=1.0, return_width=IMG_WIDTH, return_height=IMG_HEIGHT, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_height), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb

def generator_batch(data_list, nbr_classes=2, batch_size=BATCH_SIZE, return_label=True, crop_method=center_crop,
                    scale_ratio=1.0, random_scale=True, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, shuffle=True):
    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = os.path.basename(data_list[i]).split('.')
            if return_label:
                if line[0] == 'dog':
                    label = 1
                else:
                    label = 0
            img_path = data_list[i]

            if random_scale:
                scale_ratio = random.uniform(0.9, 1.1)
            img = scale_byRatio(img_path, ratio=scale_ratio, return_width=img_width,
                                crop_method=crop_method)

            X_batch[i - current_index] = img
            if return_label:
                Y_batch[i - current_index, label] = 1

        X_batch = X_batch.astype(np.float32)
        X_batch = X_batch / 255.0

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch

if __name__ == "__main__":

    print('Predicting model begins...')
    model = tf.keras.models.load_model('dog-cat_keras_load-epoch=0019-loss=0.0000-acc=1.0000-val_loss=0.8207-val_acc=0.9732.h5')

    images_path = 'test1'
    val_images = os.listdir(images_path)
    val_images.sort(key=lambda x: (int(x.split('.')[0])))
    val_images = [os.path.join(images_path, i) for i in val_images]

    nbr_imags = len(val_images)

    steps_per_epoch = int(math.ceil(nbr_imags * 1. / BATCH_SIZE))

    predict = model.predict_generator(generator_batch(val_images, return_label=False, random_scale=False, shuffle=False),
                steps=steps_per_epoch)
    test_df = pd.DataFrame({
        'id': [os.path.basename(i).split('.')[0] for i in val_images],
        'label': predict[:, 1]
    })
    test_df.to_csv('submission.csv', index=False)
