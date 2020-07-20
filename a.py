# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as KTF
import os
import cv2
import math
import random

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

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

    print('Training model begins...')

    input_ = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output_ = tf.keras.layers.Dense(2, activation='softmax', name='prediction')(x)

    model = tf.keras.models.Model(input_, output_)
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    best_model_path = './dog-cat_keras_load-epoch={epoch:04d}-loss={loss:.4f}-acc={acc:.4f}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.h5'
    best_model = tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True)

    early_Stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', verbose=1, factor=0.5, patience=3, min_lr=0.00001)

    tsboard_path = './tsboard_logs'
    if not os.path.exists(tsboard_path):
        os.mkdir(tsboard_path)
    tsboard = tf.keras.callbacks.TensorBoard(log_dir=tsboard_path, batch_size=BATCH_SIZE, update_freq='epoch',
                            histogram_freq=1, write_grads=True)

    images_path = 'train'
    images = os.listdir(images_path)
    images = [os.path.join(images_path, i) for i in images]

    random.shuffle(images)
    nbr_imags = len(images)
    nbr_train = int(math.ceil(nbr_imags * 0.8))
    nbr_val = nbr_imags - nbr_train
    train_images = images[:nbr_train]
    val_images = images[nbr_train:]

    steps_per_epoch = int(math.ceil(nbr_train * 1. / BATCH_SIZE))
    validation_steps = int(math.ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(generator_batch(train_images), steps_per_epoch=steps_per_epoch, epochs=100, verbose=1,
            validation_data=generator_batch(val_images, random_scale=False, shuffle=False),
            validation_steps=validation_steps, callbacks=[best_model, early_Stop, reduce_lr, tsboard])
