# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.backend import tensorflow_backend as KTF
import pickle

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

input_ = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu')(input_)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPool2D()(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
output_ = Dense(10, activation='softmax', name='prediction_num')(x)

model = Model(input_, output_)
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

best_model_path = './models/mnist_keras_load-epoch={epoch:04d}-loss={loss:.4f}-acc={acc:.4f}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.h5'
best_model = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True)

early_Stop = EarlyStopping(monitor='val_acc', patience=15, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', verbose=1, factor=0.5, patience=3, min_lr=0.00001)

tsboard_path = './models/logs'
tsboard = TensorBoard(log_dir=tsboard_path, batch_size=128, update_freq='epoch',
                        histogram_freq=1, write_grads=True)

model.fit(X_train, y_train, batch_size=128, epochs=100, 
          validation_data=(X_test, y_test), verbose=1, 
          shuffle=True, callbacks=[best_model, early_Stop, reduce_lr, tsboard])
