import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed
from random import randint
import tensorflow as tf
from PIL import Image, ImageChops, ImageDraw, ImageOps
from pathlib import Path
from typing import Tuple
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras_ocr
import matplotlib.pyplot as plt
import pytesseract
import shutil

import pathlib
import pandas as pd
from IPython.core.display import HTML

from src.sormani.newspaper import Newspaper
from src.sormani.sormani import Sormani
from src.sormani.system import STORAGE_DL, STORAGE_BASE, IMAGE_ROOT

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
    # self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
    #                                                                       shuffle=True,
    #                                                                       batch_size=BATCH_SIZE,
    #                                                                       image_size=IMG_SIZE)
    self.class_names = self.train_dataset.class_names
    self.file_paths =  self.train_dataset.file_paths
    self.file_paths.sort()

  def preprocessing(self, level = 50, limit = None):
    #self.change_contrast(level, limit)
    self.eliminate_black_borders((limit))
  def change_contrast(self, level = 50, limit = None):
    for i, file in enumerate(self.file_paths):
      image = Image.open(file)
      image = self._change_contrast(image, level)
      image.save(file)
      if limit is not None and i >= limit - 1:
        break
  def _change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)

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

    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])

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
      name = file.split('_')
      for i in range(len(name)):
        if name[i].isdigit():
          break
      name = ' '.join(name[:i])
      new_path = os.path.join(STORAGE_DL, name, 'train')
      os.makedirs(new_path, exist_ok=True)
      shutil.copyfile(os.path.join(filedir, file), os.path.join(new_path, file))
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
  for name in names:
    names = _move_to_train(names, os.path.join(STORAGE_DL, name, 'train'))
    names = _move_to_train(names, os.path.join(STORAGE_DL, name, 'validation'))
    names = _move_to_train(names, os.path.join(STORAGE_DL, name, 'test'))
  _move_to_train(names, os.path.join(STORAGE_BASE, 'numbers'))
  return names

def _move_to_class(current_path, name):
  for filedir, dirs, files in os.walk(current_path):
    if filedir.split('/')[-1] == name:
      for file in files:
        #p = (''.join(' ' if not ch.isdigit() else ch for ch in file).strip()).split()[-1]
        p = Path(file).stem.split('_')[-1]
        _current_path = os.path.join(current_path, p)
        os.makedirs(_current_path, exist_ok=True)
        os.rename(os.path.join(filedir, file), os.path.join(_current_path, file))

def move_to_class(name):
  _move_to_class(os.path.join(STORAGE_DL, name, 'train'), 'train')
  _move_to_class(os.path.join(STORAGE_DL, name, 'validation'), 'validation')
  _move_to_class(os.path.join(STORAGE_DL, name, 'test'), 'test')

def count_tiff():
  count = 0
  for filedir, dirs, files in os.walk(os.path.join(IMAGE_ROOT, 'TIFF')):
    count += len(files)
  return count

def set_GPUs():
  from numba import cuda
  device = cuda.get_current_device()
  device.reset()
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try: # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
      print(e)
      exit(0)

def reduce_images(name):
  image_path = os.path.join(STORAGE_DL, name, 'train')
  filedir, dirs, files = next(os.walk(image_path))
  for file in files:
    image = Image.open(os.path.join(filedir, file))
    w, h = image.size
    image = image.resize((640, 640), Image.Resampling.LANCZOS)
    image.save(os.path.join(filedir, file))
  pass

def convert_images_to_rgb(name):
  image_path = os.path.join(STORAGE_DL, name, 'train')
  filedir, dirs, files = next(os.walk(image_path))
  for file in files:
    image = Image.open(os.path.join(filedir, file))
    image = image.convert('RGB')
    image.save(os.path.join(filedir, file))
  pass

def keras_ocr_test(name):
  pipeline = keras_ocr.pipeline.Pipeline()
  image_path = os.path.join(STORAGE_DL, name, 'train')
  filedir, dirs, files = next(os.walk(image_path))
  images = []
  for file in files:
    images.append(os.path.join(filedir, file))
  images = [keras_ocr.tools.read(img) for img in images]
  prediction_groups = pipeline.recognize(images)
  pass

def tesserat_ocr(name, oem = 3, dpi = 100):
  image_path = os.path.join(STORAGE_DL, name, 'train', '02')
  filedir, dirs, files = next(os.walk(image_path))
  custom_config = r'-l ita --oem ' + str(oem) + ' --psm 4 --dpi ' + str(dpi)
  for file in files:
    files.sort()
    image = Image.open(os.path.join(filedir, file))
    text = pytesseract.image_to_string(image, config=custom_config)
    text = ''.join([n for n in text if n.isdigit()])
    print(text, ' : ', file)

def _change_contrast_PIL(img, level):
  factor = (259 * (level + 255)) / (255 * (259 - level))
  def contrast(c):
    return 128 + factor * (c - 128)
  return img.point(contrast)

def change_contrast_PIL(filedir, file, level):
  image = Image.open(os.path.join(filedir, file))
  image = _change_contrast_PIL(image, level)
  image.save(os.path.join(filedir, 'temp.tif'))

def change_contrast(img):
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  l_channel, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
  cl = clahe.apply(l_channel)
  limg = cv2.merge((cl, a, b))
  enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  return enhanced_img

def cv2_resize(img, scale):
  width = int(img.shape[1] * scale / 100)
  height = int(img.shape[0] * scale / 100)
  dim = (width, height)
  img = cv2.resize(img, dim)
  return img
def delete_non_black(img):
  min = np.array([20, 20, 20], np.uint8)
  max = np.array([255, 255, 255], np.uint8)
  mask = cv2.inRange(img, min, max)
  img[mask > 0] = [255, 255, 255]
  return img

def set_pages_numbers(name):
  def _get_file_name(file):
    file_name = Path(file).stem
    f = file_name.split('_')
    file_name_cutted = '_'.join(f[:-1])
    n = f[-2][1:]
    r = []
    ok = False
    for i in range(len(n)):
      if n[i] != '0':
        ok = True
      if ok:
        r.append(n[i])
    return file_name, file_name_cutted, np.array(r)
  image_path = os.path.join(STORAGE_BASE, 'numbers_' + name.lower().replace(' ', '_'))
  filedir, dirs, files = next(os.walk(image_path))
  files.sort()
  i = 0
  file_name_before = None
  n_before = 0
  files = np.array(files)
  Path(os.path.join(STORAGE_BASE, 'numbers')).mkdir(parents=True, exist_ok=True)
  while(i < len(files)):
    file_name, file_name_cutted, ns = _get_file_name(files[i])
    n = ns[0]
    if file_name_before is not None and file_name_before == file_name_cutted:
      n_before += 1
      if n_before < len(ns):
        n = ns[n_before]
      else:
        i += 1
        continue
    else:
      n_before = 0
    file_name_before = file_name_cutted
    file_name = '_'.join(file_name.split('_')[:-1]) + '_' + ('00' + str(n_before))[-3:] + '_' + str(n)
    shutil.copyfile(os.path.join(filedir, files[i]), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.tif')
    i += 1

def set_X(name, jump = 5):
  image_path = os.path.join(STORAGE_BASE, 'no_numbers_' + name.lower().replace(' ', '_'))
  filedir, dirs, files = next(os.walk(image_path))
  files.sort()
  Path(os.path.join(STORAGE_BASE, 'numbers')).mkdir(parents=True, exist_ok=True)
  for i, file in enumerate(files):
    if i % jump != 0:
      continue
    file_name = Path(file).stem
    # file_name = '_'.join(file_name.split('_')[:-1])
    file_name = '_'.join(file_name.split('_')[:-1]) + '_000_X'
    shutil.copyfile(os.path.join(filedir, file), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.tif')
    #os.rename(os.path.join(filedir, file), os.path.join(filedir, file_name) + '.tif')

def prepare_cnn():
  names = move_to_train()
  for name in names:
    distribute_cnn(name)
    move_to_class(name)

def get_max_box(name):
  filedir, dirs, files = next(os.walk(os.path.join(STORAGE_BASE, 'no_numbers_' + name.lower().replace(' ', '_'))))
  min_w = None
  max_w = None
  min_h = None
  max_h = None
  min_ts = None
  max_ts = None
  for file in files:
    image = Image.open(os.path.join(filedir, file))
    w, h = image.size
    if w > 1000:
      continue
    ts = np.asarray(image).mean()
    min_w = min_w if min_w is not None and min_w < w else w
    max_w = max_w if max_w is not None and max_w > w else w
    min_h = min_h if min_h is not None and min_h < h else h
    max_h = max_h if max_h is not None and max_h > h else h
    min_ts = min_ts if min_ts is not None and min_ts < ts else ts
    max_ts = max_ts if max_ts is not None and max_ts > ts else ts
  print(min_w, max_w, min_h, max_h, min_ts, max_ts)

def save_page_numbers(name):
  sormani = Sormani(name, year=2016, months=2, days=None)
  sormani.get_pages_numbers(no_resize=True, filedir = os.path.join(STORAGE_BASE, 'tmp'))

# set_pages_numbers()
# set_X()

set_GPUs()


# save_page_numbers('Avvenire')
# set_X("Avvenire")
# set_X("La Stampa")
# set_X("Il Manifesto")

prepare_cnn()
