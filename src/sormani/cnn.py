import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random
import random
import datetime
import time
import shutil

from src.sormani.newspaper import Newspaper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageTk
import tkinter as tk
from tkinter import Label, Button
from pathlib import Path
from tensorflow import keras

from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import mnist

from keras.applications.inception_v3 import *
from keras.applications.efficientnet_v2 import *
from keras.applications.inception_resnet_v2 import *
from keras.applications.convnext import *
from keras.applications.nasnet import *
from keras.applications.densenet import *
from numba import cuda


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


from src.sormani.sormani import Sormani
from src.sormani.system import STORAGE_DL, STORAGE_BASE, IMAGE_ROOT, REPOSITORY, NEWSPAPERS, IMAGE_PATH, \
  NUMBER_IMAGE_SIZE, JPG_PDF_PATH, STORAGE_BOBINE

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

class CNN:

  def __init__(self, name = 'all'):
    name = name.lower().replace(' ', '_')
    self.train_dir = os.path.join(STORAGE_DL, name)
    self.test_dir = os.path.join(STORAGE_DL, 'test')
    self.train_ds = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                                validation_split=0.2,
                                                                subset="training",
                                                                # color_mode = "grayscale",
                                                                seed=123,
                                                                shuffle=True,
                                                                image_size=IMG_SIZE,
                                                                batch_size=BATCH_SIZE)
    self.val_ds = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              # color_mode="grayscale",
                                                              seed=123,
                                                              shuffle=True,
                                                              image_size=IMG_SIZE,
                                                              batch_size=BATCH_SIZE)
    # self.test_ds = tf.keras.utils.image_dataset_from_directory(self.test_dir,
    #                                                            # color_mode="grayscale",
    #                                                            shuffle=False,
    #                                                            image_size=IMG_SIZE,
    #                                                            batch_size=BATCH_SIZE)
    self.class_names = self.train_ds.class_names

  def create_model_cnn(self, num_classes = 11, type = 'SimpleCNN', name=None):
    data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.2),
    ])
    if type == 'DenseNet201':
      base_model = DenseNet201(weights='imagenet', include_top=False)
    elif type == 'InceptionResNetV2':
      base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    elif type == 'EfficientNetV2M':
      base_model = EfficientNetV2M(weights='imagenet', include_top=False)
    elif type == 'EfficientNetV2L':
      base_model = EfficientNetV2L(weights='imagenet', include_top=False)
    else:
      model = tf.keras.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
      ])
      return model, 'SimpleCNN'
    for layer in base_model.layers:
      layer.trainable = True
    model = tf.keras.Sequential([data_augmentation,
                                 base_model,
                                 GlobalAveragePooling2D(),
                                 Dense(1024, activation='relu'),
                                 Dense(num_classes, activation='softmax')])
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    # predictions = Dense(num_classes, activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)
    return model, type if name is None else name
  def exec_cnn(self, name = None, epochs = 100):
    def process(image, label):
      image = tf.cast(image / 255., tf.float32)
      return image, label

    self.train_ds = self.train_ds.map(process)
    self.val_ds = self.val_ds.map(process)
    # self.test_ds = self.test_ds.map(process)
    if name is not None:
      name = name.lower().replace(' ', '_')
    else:
      name = ''
    # model, model_name = self.create_model_cnn(num_classes=len(self.class_names), type = 'SimpleCNN')
    model, model_name = self.create_model_cnn(num_classes=len(self.class_names), type='DenseNet201', name='DenseNet201_3')
    # model, model_name = self.create_model_cnn(num_classes=len(self.class_names), type='EfficientNetV2L')
    Path(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name)).mkdir(parents=True, exist_ok=True)
    # model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name))
    # try:
    #   file = open(os.path.join(STORAGE_BASE, 'models', name, 'best_model_' + model_name, 'results.txt'), 'r')
    #   self.val_sparse_categorical_accuracy = float(file.read())
    # finally:
    #   try:
    #     file.close()
    #   except:
    #     pass
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    callbacks = []
    callbacks.append(customCallback(name, model, model_name))
    model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs,
      callbacks=callbacks
    )
    tf.keras.models.save_model(model, os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name), save_format = 'tf')
    # file = open(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name, 'results.txt'), 'w')
    # file.write(str(self.val_sparse_categorical_accuracy))
    # file.close()
    pass

  def prediction_cnn(self):
    def process(image, label):
      image = tf.cast(image / 255., tf.float32)
      return image, label

    self.test_ds = self.test_ds.map(process)
    model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, 'best_model'))
    print("Evaluate on test data")
    results = model.evaluate(self.test_ds, batch_size=128)
    print("test loss, test acc:", results)
    predictions = model.predict(self.test_ds)
    final_prediction = np.argmax(predictions, axis=-1)
    pass

  def prediction_images_cnn(self):
    dataset = []
    for filedir, dirs, files in os.walk(self.test_dir):
      files.sort()
      for file in files:
        image = Image.open(os.path.join(filedir, file)).resize(IMG_SIZE)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        dataset.append(image)
    model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, 'best_model_1.0_0.9948'))
    prediction = np.argmax(model.predict(np.array(dataset)), axis=-1)
    pass

  def evaluate(self):
    print("Evaluate on test data")
    results = model.evaluate(self.test_ds, batch_size=128)
    print("test loss, test acc:", results)
    print("Generate predictions for 3 samples")
    predictions = model.predict(self.test_ds)
    print("predictions shape:", predictions.shape)
    final_prediction = np.argmax(predictions, axis=-1)

class customCallback(keras.callbacks.Callback):
  def __init__(self, name, model, model_name):
    self.name = name
    self.model = model
    self.model_name = model_name
    self.val_sparse_categorical_accuracy = None
  def on_epoch_end(self, epoch, logs=None):
    logs_val = logs['val_sparse_categorical_accuracy']
    min = 0.95
    if logs_val > min and (self.val_sparse_categorical_accuracy is None or logs_val > self.val_sparse_categorical_accuracy) or logs_val == 1.0:
      model_path = os.path.join(STORAGE_BASE, 'models', self.name, 'best_model_' + self.model_name)
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy improved from {self.val_sparse_categorical_accuracy} to'
            f' {logs_val}, saving model to {model_path}')
      self.val_sparse_categorical_accuracy = logs_val
      tf.keras.models.save_model(self.model, model_path, save_format='tf')
      try:
        file = open(os.path.join(STORAGE_BASE, 'models', self.name, 'best_model_' + self.model_name, 'results.txt'), 'w')
        file.write(str(self.val_sparse_categorical_accuracy))
      finally:
        try:
          file.close()
        except:
          pass
    elif logs_val <= min:
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy equal to {logs_val}'
            f' is lower than the minimum value for saving')
    else:
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy equal to {logs_val}'
            f' did not improve from {self.val_sparse_categorical_accuracy}')


ns = 'La Domenica del Corriere'

cnn = CNN(ns)
cnn.exec_cnn(ns, epochs = 100)
