# Libraries
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from glob import glob
from keras.callbacks import ModelCheckpoint
from zipfile import ZipFile


# Extracting Data From Zip
with ZipFile('flood.zip') as file:
    file.extractall('dataset/')


# Hyperparameters
BATCH_SIZE = 3
EPOCHS = 5
SPLIT = 0.20
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# Data Preprocessing
X = []
Y = []

data_path = 'dataset'

classes = os.listdir(data_path)
classes = classes[:2]

for i, name in enumerate(classes):
    if name == 'Image':
        images = glob(f'{data_path}/{name}/*')

        for image in images:
            img = cv2.imread(image)
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    
    elif name == 'Mask':

        masks = glob(f'{data_path}/{name}/*.jpg')

        for mask in masks:
            file_type = '.jpg'
            os.rename(mask, (mask[:-4] + file_type))

        for mask in masks:
            mas = cv2.imread(mask)
            mas = cv2.cvtColor(mas, cv2.COLOR_RGB2GRAY)
            mas = cv2.cvtColor(mas, cv2.COLOR_GRAY2RGB)
            Y.append(cv2.resize(mas, (IMG_SIZE, IMG_SIZE)))

X = np.asarray(X)
Y = np.asarray(Y)

X = X.astype('float32')
X = Y.astype('float32')

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    shuffle= True,
                                                    random_state= 24)


# Creating Model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same',),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(256, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding= 'same'),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2DTranspose(64, (2, 2), strides= (2, 2), padding= 'same'),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation= 'relu', kernel_initializer= 'he_normal', padding= 'same'),
    layers.Conv2D(3, (1, 1), activation='sigmoid')

])

model.compile(optimizer= 'adam',
              loss = 'categorical_crossentropy',
              metrics= ['accuracy']
)


# Model Checkpoint
check = ModelCheckpoint('output/checkpoint.h5',
                        monitor= 'val_accuracy',
                        verbose= 1,
                        save_best_only= True,
                        save_weights_only= True
                        )


# Training Model
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
model.fit(X_train,Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          callbacks= check,
          validation_data= (X_test, Y_test),
          verbose= 1
          )