import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed
from random import randint
import tensorflow as tf
from PIL import Image
from PIL import ImageChops
from pathlib import Path
from typing import Tuple
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np

import pathlib
import pandas as pd
from IPython.core.display import HTML

from src.sormani.system import STORAGE_DL

BATCH_SIZE = 32
IMG_SIZE = (7500, 600)

class cnn:

  def __init__(self, name):
    self.train_dir = os.path.join(STORAGE_DL, name, 'train')
    self.validation_dir = os.path.join(STORAGE_DL, name, 'validation')
    self.train_dataset = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                                     shuffle=True,
                                                                     batch_size=BATCH_SIZE,
                                                                     image_size=IMG_SIZE)
    self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
                                                                          shuffle=True,
                                                                          batch_size=BATCH_SIZE,
                                                                          image_size=IMG_SIZE)
    self.class_names = self.train_dataset.class_names

  def exec_cnn(self):
    val_batches = tf.data.experimental.cardinality(self.validation_dataset)
    test_dataset = self.validation_dataset.take(val_batches // 5)
    self.validation_dataset = self.validation_dataset.skip(val_batches // 5)

    print('Number of validation batches: %d' % tf.data.experimental.cardinality(self.validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    AUTOTUNE = tf.data.AUTOTUNE

    self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
    self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # data_augmentation = tf.keras.Sequential([
    #   tf.keras.layers.RandomFlip('horizontal'),
    #   tf.keras.layers.RandomRotation(0.2),
    # ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(self.train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    # Let's take a look at the base model architecture
    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    initial_epochs = 10

    loss0, accuracy0 = model.evaluate(self.validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(self.train_dataset,
                        epochs=initial_epochs,
                        validation_data=self.validation_dataset)


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    base_model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                  metrics=['accuracy'])

    model.summary()

    len(model.trainable_variables)

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(self.train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=self.validation_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    plt.figure(figsize=(10, 10))
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(image_batch[i].astype("uint8"))
      plt.title(self.class_names[predictions[i]])
      plt.axis("off")

def prepare_png(name):
  image_path = os.path.join(STORAGE_DL, name, 'all')
  train_path = os.path.join(STORAGE_DL, name, 'train')
  for filedir, dirs, files in os.walk(image_path):
    files.sort()
    for file in files:
      image = Image.open(os.path.join(image_path, file))
      file_name = Path(file).stem + '.png'
      image.save(os.path.join(train_path, file_name), 'PNG', quality=100)
  crop_png(name)

def crop_png(name):
  image_path = os.path.join(STORAGE_DL, name, 'train')
  for filedir, dirs, files in os.walk(image_path):
    files.sort()
    for file in files:
      image = Image.open(os.path.join(image_path, file))
      w,h = image.size
      image1 = image.crop((200, 0, 800, h))
      image2 = image.crop((w - 800, 0, w - 200, h))
      image = Image.new('RGB', (1200, h))
      image.paste(image1, (0, 0))
      image.paste(image2, (600, 0))
      image.save(os.path.join(image_path, file), 'PNG', quality=100)
    break
  pass


def distribute_cnn(name, validation = 0.1, test = 0.1):
  seed(28362)
  train_path = os.path.join(STORAGE_DL, name, 'train')
  os.makedirs(train_path, exist_ok=True)
  validation_path = os.path.join(STORAGE_DL, name, 'validation')
  os.makedirs(validation_path, exist_ok=True)
  test_path = os.path.join(STORAGE_DL, name, 'test')
  os.makedirs(test_path, exist_ok=True)
  for filedir, dirs, files in os.walk(train_path):
    l = len(files)
    for i in range(int(l * validation)):
      j = randint(0, len(files) - 1)
      file = files[j]
      os.rename(os.path.join(train_path, file), os.path.join(validation_path, file))
      files.remove(file)
    for i in range(int(l * test)):
      j = randint(0, len(files) - 1)
      file = files[j]
      os.rename(os.path.join(train_path, file), os.path.join(test_path, file))
      files.remove(file)
  pass

def _move_to_train(names, current_path):
  for filedir, dirs, files in os.walk(current_path):
    for file in files:
      name = file.split('_')[1:]
      for i in range(len(name)):
        if name[i].isdigit():
          break
      name = ' '.join(name[:i])
      new_path = os.path.join(STORAGE_DL, name, 'train')
      os.makedirs(new_path, exist_ok=True)
      os.rename(os.path.join(filedir, file), os.path.join(new_path, file))
      if not name in names:
        names.append(name)
  to_delete = []
  for filedir, dirs, files in os.walk(current_path):
    if not len(files) and not len(dirs):
      to_delete.append(filedir)
  for dir in to_delete:
    os.rmdir(dir)
  return names
def move_to_train():
  names = list(next(os.walk(STORAGE_DL)))[1]
  names.remove('all')
  for name in names:
    names = _move_to_train(names, os.path.join(name, STORAGE_DL, name, 'train'))
    names = _move_to_train(names, os.path.join(name, STORAGE_DL, name, 'validation'))
    names = _move_to_train(names, os.path.join(name, STORAGE_DL, name, 'test'))
  return names

def _move_to_class(current_path, name):
  for filedir, dirs, files in os.walk(current_path):
    if filedir.split('/')[-1] == name:
      for file in files:
        p = (''.join(' ' if not ch.isdigit() else ch for ch in file).strip()).split()[-1]
        _current_path = os.path.join(current_path, p)
        os.makedirs(_current_path, exist_ok=True)
        os.rename(os.path.join(filedir, file), os.path.join(_current_path, file))

def move_to_class(name):
  _move_to_class(os.path.join(STORAGE_DL, name, 'train'), 'train')
  _move_to_class(os.path.join(STORAGE_DL, name, 'validation'), 'validation')
  _move_to_class(os.path.join(STORAGE_DL, name, 'test'), 'test')

def prepare_cnn():
  names = move_to_train()
  # for name in names:
  #   distribute_cnn(name)
  #   move_to_class(name)

# prepare_cnn()

# cnn = cnn("La Stampa")
# cnn.exec_cnn()

crop_png("La Stampa")