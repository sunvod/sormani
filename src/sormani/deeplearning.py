import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random
import random
import datetime
import time

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

  def create_model_cnn(self, num_classes = 11, type = 'SimpleCNN'):
    if type == 'DenseNet201':
      base_model = DenseNet201(weights='imagenet', include_top=False)
    elif type == 'InceptionResNetV2':
      base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    elif type == 'EfficientNetV2M':
      base_model = EfficientNetV2M(weights='imagenet', include_top=False)
    else:
      model = tf.keras.Sequential([
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
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, type
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
    model, model_name = self.create_model_cnn(num_classes=len(self.class_names), type = 'DenseNet201')
    Path(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name)).mkdir(parents=True, exist_ok=True)
    # model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name))
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
    if logs_val > min and (self.val_sparse_categorical_accuracy is None or logs_val > self.val_sparse_categorical_accuracy):
      model_path = os.path.join(STORAGE_BASE, 'models', self.name, 'best_model_' + self.model_name)
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy improved from {self.val_sparse_categorical_accuracy} to'
            f' {logs_val}, saving model to {model_path}')
      self.val_sparse_categorical_accuracy = logs_val
      tf.keras.models.save_model(self.model, model_path, save_format='tf')
    elif logs_val <= min:
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy equal to {logs_val}'
            f' is lower than the minimum value for saving')
    else:
      print(f'\nEpoch {epoch + 1}: val_sparse_categorical_accuracy equal to {logs_val}'
            f' did not improve from {self.val_sparse_categorical_accuracy}')

def distribute_cnn(name, validation = 0.0, test = 0.1):
  seed(28362)
  train_path = os.path.join(STORAGE_DL, name, 'train')
  os.makedirs(train_path, exist_ok=True)
  validation_path = os.path.join(STORAGE_DL, name, 'validation')
  os.makedirs(validation_path, exist_ok=True)
  test_path = os.path.join(STORAGE_DL, name, 'test')
  os.makedirs(test_path, exist_ok=True)
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    l = len(files)
    for i in range(int(l * validation)):
      j = random.randint(0, len(files) - 1)
      file = files[j]
      os.rename(os.path.join(filedir, file), os.path.join(validation_path, file))
      files.remove(file)
    for i in range(int(l * test)):
      j = random.randint(0, len(files) - 1)
      file = files[j]
      os.rename(os.path.join(filedir, file), os.path.join(test_path, file))
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
      # shutil.copyfile(os.path.join(filedir, file), os.path.join(new_path, file))
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
  # _move_to_class(os.path.join(STORAGE_DL, name, 'validation'), 'validation')
  # _move_to_class(os.path.join(STORAGE_DL, name, 'test'), 'test')

def count_tiff():
  count = 0
  tot = 0
  for newspaper in NEWSPAPERS:
    count = 0
    for filedir, dirs, files in os.walk(os.path.join(IMAGE_ROOT, 'TIFF', newspaper)):
      count += len(files)
      if not len(files):
        pass
    if count:
      print(f'{newspaper}: {count}')
      tot += count
  print(f'Totale: {tot}')
  return count

def set_GPUs():
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

# ----------------------------------------------------------------
#   Aggiunge il numero al nome del file
#     * il file di input deve essere in numbers_<nome giornale>
#     * il file di output va in numbers
# ----------------------------------------------------------------
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
    os.rename(os.path.join(filedir, files[i]), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.jpg')
    i += 1

def set_X(name, jump = 1):
  image_path = os.path.join(STORAGE_BASE, 'no_numbers_' + name.lower().replace(' ', '_'))
  filedir, dirs, files = next(os.walk(image_path))
  files.sort()
  Path(os.path.join(STORAGE_BASE, 'no_numbers')).mkdir(parents=True, exist_ok=True)
  for i, file in enumerate(files):
    if i % jump != 0:
      continue
    file_name = Path(file).stem
    file_name = '_'.join(file_name.split('_')[:-3]) + '_' + str(i) + '_X'
    os.rename(os.path.join(filedir, file), os.path.join(STORAGE_BASE, 'no_numbers', file_name) + '.jpg')

def prepare_cnn():
  names = move_to_train()
  for name in names:
    if name == 'train':
      continue
    distribute_cnn(name)
    move_to_class(name)

def put_all():
  names = move_to_train()
  for name in names:
    distribute_cnn(name)
    move_to_class(name)
def change_newspaper_name(newspaper_name, oldname, newname):
  for filedir, dirs, files in os.walk(os.path.join(IMAGE_ROOT, IMAGE_PATH, newspaper_name)):
    files.sort()
    for file in files:
      oldname = oldname.replace(' ', '_')
      newname = newname.replace(' ', '_')
      newfile = file.replace(oldname, newname)
      os.rename(os.path.join(filedir, file), os.path.join(filedir, newfile))

def convert_to_jpg(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      if Path(os.path.join(filedir, file)).suffix != '.jpg':
        image = Image.open(os.path.join(filedir, file))
        os.remove(os.path.join(filedir, file))
        image.save(os.path.join(filedir, Path(file).stem + '.jpg'), format="jpeg")

def standardize_dimension(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file))
      image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
      image.save(os.path.join(filedir, file))

def to_rgb(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file))
      image = image.convert('RGB')
      image.save(os.path.join(filedir, file))
def to_gray(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file)).convert('L')
      image.save(os.path.join(filedir, file))
def to_base(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE,  name.lower().replace(' ', '_'))):
    for file in files:
      os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_BASE,  name.lower().replace(' ', '_')), file))

def to_2_classes(name = 'train'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      p = Path(file).stem.split('_')[-1]
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'X'), exist_ok=True)
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'N'), exist_ok=True)
      if p == 'X':
        os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'X', file))
      else:
        os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'N', file))

def to_10_classes(name = 'all'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'numbers')):
    for file in files:
      n = Path(file).stem.split('_')[-1]
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(n)), exist_ok=True)
      os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), str(n), file))

def to_X(name = 'all'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'no_numbers')):
    for file in files:
      n = Path(file).stem.split('_')[-1]
      if n == 'X':
        os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(n)), exist_ok=True)
        os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'X', file))
def force_to_X():
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'no_numbers')):
    for file in files:
      new_file = '_'.join(file.split('_')[:-1]) + '_X.jpg'
      os.rename(os.path.join(filedir, file), os.path.join(filedir, new_file))
def to_n_classes(name = 'all', n = 11, source = None, resize = False):
  name = name.lower().replace(' ', '_')
  if source is None:
    sources = [os.path.join(STORAGE_BASE, REPOSITORY, name, 'sure', 'numbers'),
               os.path.join(STORAGE_BASE, REPOSITORY, name, 'sure', 'no_numbers'),
               os.path.join(STORAGE_BASE, REPOSITORY, name, 'notsure', 'numbers'),
               os.path.join(STORAGE_BASE, REPOSITORY, name, 'notsure', 'no_numbers'),
               os.path.join(STORAGE_BASE, 'numbers')]
  elif isinstance(source, str):
    sources = [source]
  else:
    sources = source
  for i in range(n - 1):
    os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(i)), exist_ok=True)
  os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'X'), exist_ok=True)
  for source in sources:
    for filedir, dirs, files in os.walk(source):
      for file in files:
        n = Path(file).stem.split('_')[-1]
        image = Image.open(os.path.join(filedir, file))
        if resize:
          image = image.resize(NUMBER_IMAGE_SIZE, Image.Resampling.LANCZOS)
        os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(n)), exist_ok=True)
        image.save(os.path.join(os.path.join(STORAGE_DL, name), str(n), file))
        # os.remove(os.path.join(filedir, file))
def move_to_test(name = 'numbers'):
  dest_path = os.path.join(STORAGE_BASE, 'test/images')
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'repository', 'sure', 'X')):
    for file in files:
      os.rename(os.path.join(filedir, file), os.path.join(STORAGE_DL, dest_path, file))
def get_max_box(name = None):
  if name is not None:
    # filedir, dirs, files = next(os.walk(os.path.join(STORAGE_BASE, 'no_numbers_' + name.lower().replace(' ', '_'))))
    pass
  filedir, dirs, files = next(os.walk(os.path.join(STORAGE_BASE, 'numbers')))
  min_w = None
  max_w = None
  min_h = None
  max_h = None
  min_mean = None
  max_mean = None
  for file in files:
    img = cv2.imread(os.path.join(filedir, file))
    h, w, _ = img.shape
    if w > 1000:
      continue
    ts = np.asarray(img).mean()
    min_w = min_w if min_w is not None and min_w < w else w
    max_w = max_w if max_w is not None and max_w > w else w
    min_h = min_h if min_h is not None and min_h < h else h
    max_h = max_h if max_h is not None and max_h > h else h
    min_mean = min_mean if min_mean is not None and min_mean < ts else ts
    max_mean = max_mean if max_mean is not None and max_mean > ts else ts
  print(f'Larghezza (min , max): {min_w} {max_w}')
  print(f'Altezza (min max): {min_h} {max_h}')
  print(f'Media (min , max): {round(min_mean, 3)} , {round(max_mean, 3)}')

def save_page_numbers(name):
  sormani = Sormani(name, year=2016, months=2, days=None)
  sormani.get_pages_numbers(no_resize=True, filedir = os.path.join(STORAGE_BASE, 'tmp'))

def open_win_rename_images_files(count, filedir, file):
  def close():
    gui.destroy()
    exit()
  def number_chosen(button_press, filedir, file):
    if button_press != 'ok':
      for i in range(4 if button_press == '4X' else 1):
        file_name = Path(file[i]).stem
        n = file_name.split('_')[-1]
        if len(n) == 1:
          file_name = '_'.join(file_name.split('_')[:-1])
        new_file = file_name + '_' + str(button_press[-1]) + '.jpg'
        os.rename(os.path.join(filedir, file[i]), os.path.join(STORAGE_BASE, 'numbers', new_file))
    else:
      new_file = file
      os.rename(os.path.join(filedir, file[0]), os.path.join(STORAGE_BASE, 'numbers', new_file))
    gui.destroy()
  gui = tk.Tk()
  gui.title('')
  w = 1300  # Width
  h = 500  # Height
  screen_width = gui.winfo_screenwidth()  # Width of the screen
  screen_height = gui.winfo_screenheight()  # Height of the screen
  # Calculate Starting X and Y coordinates for Window
  x = (screen_width / 2) - (w / 2)
  y = (screen_height / 2) - (h / 2)
  gui.geometry('%dx%d+%d+%d' % (w, h, x, y))
  gui_frame = tk.Frame(gui)
  gui_frame.pack(fill=tk.X, side=tk.BOTTOM)
  button_frame = tk.Frame(gui_frame)
  button_frame.columnconfigure(0, weight=1)
  button_frame.pack(fill=tk.X, side=tk.BOTTOM)
  button_frame.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10,10), pady=(10,10))
  n_col = 6
  buttons = [[0 for x in range(n_col)] for x in range(3)]
  for i in range(3):
    for j in range(n_col):
      text = i * (n_col) + j
      if text == 10:
        text = 'X'
      if text == 11:
        text = 'ok'
      if text == 12:
        text = 'P'
      if text == 13:
        text = 'A'
      if text == 14:
        text = 'G'
      if text == 15:
        text = 'I'
      if text == 16:
        text = 'N'
      if text == 17:
        text = '4X'
      pixel = tk.PhotoImage(width=1, height=1)
      buttons[i][j] = tk.Button(button_frame,
                                text=text,
                                compound="center",
                                font=('Aria', 24),
                                height=2,
                                width=4,
                                padx=0,
                                pady=0,
                                command=lambda number=str(text): number_chosen(number, filedir, file))
      buttons[i][j].columnconfigure(i)
      buttons[i][j].grid(row=i, column=j, sticky=tk.W+tk.E, padx=(5, 5), pady=(5, 5))
  image = Image.open(os.path.join(filedir, file[0]))
  image = image.resize(NUMBER_IMAGE_SIZE)
  img = ImageTk.PhotoImage(image)
  image_frame = tk.Frame(gui_frame)
  image_frame.grid(row=0, column=1, sticky=tk.W + tk.E)
  label = Label(image_frame, image = img)
  label.columnconfigure(1, weight=1)
  label.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(50, 50))
  small_image_frame = tk.Frame(image_frame)
  small_image_frame.grid(row=1, column=0, sticky=tk.W + tk.E)
  if len(file) > 1:
    image_1 = Image.open(os.path.join(filedir, file[len(file) - 3]))
    image_1 = image_1.resize((95, 95))
    img_1 = ImageTk.PhotoImage(image_1)
    small_label_1 = Label(small_image_frame, image = img_1)
    small_label_1.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(3, 3), pady=(6, 6))
  if len(file) > 2:
    image_2 = Image.open(os.path.join(filedir, file[len(file) - 2]))
    image_2 = image_2.resize((95, 95))
    img_2 = ImageTk.PhotoImage(image_2)
    small_label_2 = Label(small_image_frame, image = img_2)
    small_label_2.grid(row=0, column=1, sticky=tk.W + tk.E, padx=(3, 3), pady=(6, 6))
  if len(file) > 3:
    image_3 = Image.open(os.path.join(filedir, file[len(file) - 1]))
    image_3 = image_3.resize((95, 95))
    img_3 = ImageTk.PhotoImage(image_3)
    small_label_3 = Label(small_image_frame, image = img_3)
    small_label_3.grid(row=0, column=2, sticky=tk.W + tk.E, padx=(3, 3), pady=(6, 6))
  label2 = Label(gui, text = str(count) + '. ' + file[0], font = ('Arial', 14))
  label2.pack(pady=20)
  exit_button = Button(gui_frame, text="Exit", font = ('Arial', 18), command=close)
  exit_button.columnconfigure(2, weight=1)
  exit_button.grid(row=0, column=2, sticky=tk.W + tk.E)
  gui.mainloop()

def rename_images_files(name):
  count = 1
  _files = []
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY + '_' + name.lower().replace(' ', '_'))):
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'numbers')):
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, name.lower().replace(' ', '_'), 'notsure', 'numbers')):
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, name.lower().replace(' ', '_'), 'notsure', 'no_numbers')):
    files.sort()
    i = 0
    for j, file in enumerate(files):
      if i > 3 or j == len(files) - 1:
        open_win_rename_images_files(count, filedir, np.array(_files))
        if os.path.isfile(os.path.join(filedir, _files[len(_files) - 1])):
          _files.pop(0)
        else:
          _files = []
          i = 0
      _files.append(file)
      i += 1
      count += 1

def open_win_pages_files(count, filedir, file):
  def close():
    gui.destroy()
    exit()
  def number_chosen(button_press, filedir, file):
    if button_press != 'ok':
      new_file = '_'.join(file.split('_')[:-1]) + '_' + str(button_press) + '.jpg'
      os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_BASE, 'numbers'), new_file))
    gui.destroy()

  gui = tk.Tk()
  gui.title('')
  w = 1000  # Width
  h = 500  # Height

  screen_width = gui.winfo_screenwidth()  # Width of the screen
  screen_height = gui.winfo_screenheight()  # Height of the screen

  # Calculate Starting X and Y coordinates for Window
  x = (screen_width / 2) - (w / 2)
  y = (screen_height / 2) - (h / 2)

  gui.geometry('%dx%d+%d+%d' % (w, h, x, y))
  gui_frame = tk.Frame(gui)
  gui_frame.pack(fill=tk.X, side=tk.BOTTOM)

  button_frame = tk.Frame(gui_frame)
  button_frame.columnconfigure(0, weight=1)
  button_frame.pack(fill=tk.X, side=tk.BOTTOM)
  button_frame.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10,10), pady=(10,10))

  buttons = [[0 for x in range(4)] for x in range(3)]

  for i in range(3):
    for j in range(4):
      text = i * 4 + j
      if text == 10:
        text = 'X'
      if text == 11:
        text = 'ok'
      pixel = tk.PhotoImage(width=1, height=1)
      buttons[i][j] = tk.Button(button_frame,
                                text=text,
                                compound="center",
                                font=('Aria', 24),
                                height=2,
                                width=4,
                                padx=0,
                                pady=0,
                                command=lambda number=str(text): number_chosen(number, filedir, file))
      buttons[i][j].columnconfigure(i)
      buttons[i][j].grid(row=i, column=j, sticky=tk.W+tk.E, padx=(5, 5), pady=(5, 5))
  image = Image.open(os.path.join(filedir, file))
  img = ImageTk.PhotoImage(image)
  label = Label(gui_frame, image = img)
  label.columnconfigure(1, weight=1)
  label.grid(row=0, column=1, sticky=tk.W + tk.E, padx=(50, 50))
  label2 = Label(gui, text = str(count) + '. ' + file, font = ('Arial', 14))
  label2.pack(pady=20)
  exit_button = Button(gui_frame, text="Exit", font = ('Arial', 18), command=close)
  exit_button.columnconfigure(2, weight=1)
  exit_button.grid(row=0, column=2, sticky=tk.W + tk.E)
  gui.mainloop()

def rename_pages_files(name):
  count = 1
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'numbers')):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, name.lower().replace(' ', '_'), 'notsure', 'numbers')):
    files.sort()
    for file in files:
      open_win_pages_files(count, filedir, file)
      count += 1

def delete_name(name):
  dir_name = name.lower().replace(' ', '_')
  name = name.replace(' ', '_')
  for filedir, dirs, files in os.walk(os.path.join(os.path.join(STORAGE_DL, dir_name))):
    for file in files:
      if file.find(name) == -1:
        os.remove(os.path.join(filedir, file))

def transform_images(name):
  name = name.lower().replace(' ', '_')
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    files.sort()
    for file in files:
      img = cv2.imread(os.path.join(filedir, file))
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      thresh, binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      kernel = np.ones((3, 3), np.uint8)
      binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=5)
      gray = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
      gray = 255 - gray
      cv2.imwrite(os.path.join(STORAGE_BASE, 'tmp', file), gray)

def change_ins_file_name():
  for one_root in [os.path.join(IMAGE_ROOT, IMAGE_PATH), os.path.join(IMAGE_ROOT, JPG_PDF_PATH)]:
    for filedir, dirs, files in os.walk(one_root):
      dirs.sort()
      for dir in dirs:
        _dir = dir.replace(' ', '_')
        if len(dir) > 4 and dir[:2].isdigit() and not dir[:5].isdigit():
          if dir[3:6] == 'INS':
            for fd, _, files in os.walk(os.path.join(filedir, dir)):
              files.sort()
              for file in files:
                print(file)
                if file.split('_')[-3] != 'INS':
                  new_file = '_'.join(file.split('_')[:-2]) + '_' + _dir + '_' + file.split('_')[-1]
                  os.rename(os.path.join(fd, file), os.path.join(fd, new_file))


def delete_bing():
  for filedir, dirs, files in os.walk('/mnt/storage01/sormani/TIFF/La Domenica del Corriere/1900/01/01'):
    files.sort()
    for file in files:
      if file.split('_')[-1] == 'thresh.tif' or file.split('_')[-1] == 'bing.tif':
        os.remove(os.path.join(filedir, file))

# set_GPUs()

ns = 'La Domenica del Corriere'

# cnn = CNN(ns)
# cnn.exec_cnn(ns, epochs = 100)

# count_tiff()

# change_newspaper_name('Osservatore Romano', 'Avvenire', 'Osservatore Romano')

# rename_images_files(ns)

# to_n_classes(ns, n=2, resize=True)

# delete_name('Avvenire')

# get_max_box()

# change_ins_file_name()

# force_to_X()

# set_bobine_images()

# set_bobine_merges()

# rotate_bobine_fotogrammi()


