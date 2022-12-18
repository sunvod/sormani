import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random
import random
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


import cv2
import numpy as np
import keras_ocr
import matplotlib.pyplot as plt
import pytesseract


from src.sormani.sormani import Sormani
from src.sormani.system import STORAGE_DL, STORAGE_BASE, IMAGE_ROOT, REPOSITORY, NEWSPAPERS, IMAGE_PATH
import tensorflow_datasets as tfds

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

  def create_simple_model_cnn(self, num_classes = 11):
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
    return model, 'SimpleCCN'
  def create_model_cnn(self, num_classes = 11):
    # base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    # base_model = EfficientNetV2M(weights='imagenet', include_top=False)
    base_model = DenseNet201(weights='imagenet', include_top=False)
    for layer in base_model.layers:
      layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, 'DenseNet201'
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
    num_classes = len(self.class_names)
    model, model_name = self.create_model_cnn(num_classes)
    # model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, 'models', name, 'last_model_' + model_name))
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    callbacks = []
    callbacks.append(CustomCallback())
    mcp_save = ModelCheckpoint(os.path.join(STORAGE_BASE, 'models', name, 'best_model_' + model_name),
                               verbose=1,
                               save_weights_only=False,
                               save_best_only=True,
                               monitor='val_sparse_categorical_accuracy',
                               mode='max',
                               callbacks = callbacks)
    model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=epochs,
      callbacks=[mcp_save]
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


class CustomCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    pass

# callbacks.append(CustomCallback())
    # tf.keras.models.save_model(model, STORAGE_BASE)
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
def to_11_classes(name = 'all', source = None):
  name = name.lower().replace(' ', '_')
  if source is None:
    sources = [os.path.join(STORAGE_BASE, REPOSITORY, name, 'sure', 'numbers'),
               os.path.join(STORAGE_BASE, REPOSITORY, name, 'sure', 'no_numbers'),
               # os.path.join(STORAGE_BASE, REPOSITORY, name, 'notsure', 'numbers'),
               # os.path.join(STORAGE_BASE, REPOSITORY, name, 'notsure', 'no_numbers'),
               os.path.join(STORAGE_BASE, 'numbers')]
  elif isinstance(source, str):
    sources = [source]
  else:
    sources = source
  for source in sources:
    for filedir, dirs, files in os.walk(source):
      for file in files:
        n = Path(file).stem.split('_')[-1]
        if n == 'X':
          os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'X'), exist_ok=True)
          os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'X', file))
        else:
          os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(n)), exist_ok=True)
          os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), str(n), file))
def move_to_test(name = 'numbers'):
  dest_path = os.path.join(STORAGE_BASE, 'test/images')
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'repository', 'sure', 'X')):
    for file in files:
      os.rename(os.path.join(filedir, file), os.path.join(STORAGE_DL, dest_path, file))
def get_max_box():
  # filedir, dirs, files = next(os.walk(os.path.join(STORAGE_BASE, 'no_numbers_' + name.lower().replace(' ', '_'))))
  filedir, dirs, files = next(os.walk(os.path.join(STORAGE_BASE, 'numbers')))
  min_w = None
  max_w = None
  min_h = None
  max_h = None
  min_mean = None
  max_mean = None
  min_perimeter = None
  max_perimeter = None
  min_area = None
  max_area = None
  for file in files:
    img = cv2.imread(os.path.join(filedir, file))
    h, w, _ = img.shape
    if w > 1000:
      continue
    edges = cv2.Canny(img, 1, 50)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

def open_win(count, filedir, file):
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

def rename_images_files():
  count = 1
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, 'unita', 'notsure', 'numbers')):
    files.sort()
    for file in files:
      open_win(count, filedir, file)
      count += 1

set_GPUs()

# cnn = CNN('Unita')
# cnn.exec_cnn('Unita', epochs = 100)
#
# to_11_classes('Unita')

# count_tiff()

# change_newspaper_name('Osservatore Romano', 'Avvenire', 'Osservatore Romano')

# to_11_classes('Unita')

# rename_images_files()

get_max_box()