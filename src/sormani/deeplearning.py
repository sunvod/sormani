import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random
import random
import tensorflow as tf
from PIL import Image, ImageChops, ImageDraw, ImageOps
from pathlib import Path

import cv2
import numpy as np
import keras_ocr
import matplotlib.pyplot as plt
import pytesseract


from src.sormani.sormani import Sormani
from src.sormani.system import STORAGE_DL, STORAGE_BASE, IMAGE_ROOT
import tensorflow_datasets as tfds

BATCH_SIZE = 32
IMG_SIZE = (48, 48)


class CNN:

  def __init__(self, name = 'All'):
    self.train_dir = os.path.join(STORAGE_DL, name)
    self.test_dir = os.path.join(STORAGE_DL, 'test')
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                       fname='flower_photos',
                                       untar=True)
    # self.train_dir = pathlib.Path(data_dir)
    self.train_ds = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                                validation_split=0.2,
                                                                subset="training",
                                                                color_mode = "grayscale",
                                                                seed=123,
                                                                shuffle=True,
                                                                image_size=IMG_SIZE,
                                                                batch_size=BATCH_SIZE)
    self.val_ds = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                              validation_split=0.2,
                                                              subset="validation",
                                                              color_mode="grayscale",
                                                              seed=123,
                                                              shuffle=True,
                                                              image_size=IMG_SIZE,
                                                              batch_size=BATCH_SIZE)
    self.test_ds = tf.keras.utils.image_dataset_from_directory(self.test_dir,
                                                               color_mode="grayscale",
                                                               shuffle=False,
                                                               image_size=IMG_SIZE,
                                                               batch_size=BATCH_SIZE)
    self.class_names = self.train_ds.class_names
  def exec_cnn(self):
    # plt.figure(figsize=(10, 10))
    # for images, labels in self.train_ds.take(1):
    #   for i in range(9):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title(self.class_names[labels[i]])
    #     plt.axis("off")
    # plt.show()
    # target_train = tf.keras.utils.to_categorical(self.train_ds, self.class_names)
    # target_test = tf.keras.utils.to_categorical(self.train_ds, self.class_names)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    normalized_ds = self.train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    AUTOTUNE = tf.data.AUTOTUNE
    num_classes = len(self.class_names)
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
    # model = tf.keras.models.load_model(STORAGE_BASE)
    model.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(
      self.train_ds,
      validation_data=self.val_ds,
      epochs=10
    )
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(self.test_ds, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(self.test_ds)
    print("predictions shape:", predictions.shape)
    final_prediction = np.argmax(predictions, axis=-1)
    pass
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
      j = randint(0, len(files) - 1)
      file = files[j]
      os.rename(os.path.join(filedir, file), os.path.join(validation_path, file))
      files.remove(file)
    for i in range(int(l * test)):
      j = randint(0, len(files) - 1)
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
    #shutil.copyfile(os.path.join(filedir, files[i]), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.tif')
    os.rename(os.path.join(filedir, files[i]), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.tif')
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
    # shutil.copyfile(os.path.join(filedir, file), os.path.join(STORAGE_BASE, 'numbers', file_name) + '.tif')
    os.rename(os.path.join(filedir, file), os.path.join(filedir, file_name) + '.tif')

def prepare_cnn():
  names = move_to_train()
  for name in names:
    if name == 'All':
      continue
    distribute_cnn(name)
    move_to_class(name)

def put_all():
  names = move_to_train()
  for name in names:
    distribute_cnn(name)
    move_to_class(name)

def convert_to_jpg(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file))
      os.remove(os.path.join(filedir, file))
      image.save(os.path.join(filedir, Path(file).stem + '.jpg'), format="jpeg")

def standardize_dimension(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file))
      image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
      image.save(os.path.join(filedir, file))

def to_rgb(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      image = Image.open(os.path.join(filedir, file))
      image = image.convert('RGB')
      image.save(os.path.join(filedir, file))

def to_base(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), file))

def to_2_classes(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      p = Path(file).stem.split('_')[-1]
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'X'), exist_ok=True)
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), 'N'), exist_ok=True)
      if p == 'X':
        os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'X', file))
      else:
        os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), 'N', file))

def to_10_classes(name = 'All'):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_DL, name)):
    for file in files:
      n = Path(file).stem.split('_')[-1]
      os.makedirs(os.path.join(os.path.join(STORAGE_DL, name), str(n)), exist_ok=True)
      os.rename(os.path.join(filedir, file), os.path.join(os.path.join(STORAGE_DL, name), str(n), file))

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


set_GPUs()

cnn = CNN()
cnn.exec_cnn()

