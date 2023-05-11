import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random
import random
import datetime
import time
import shutil
import csv
from datetime import datetime as dt
from PyPDF2 import PdfWriter, PdfReader
import aspose.words as aw
from PyPDF2 import PdfWriter, PdfReader
from pdf2image import convert_from_path

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
  sormani = Sormani(name, years=2016, months=2, days=None)
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
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY + '_' + name.lower().replace(' ', '_'))):
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, 'numbers')):
  for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, name.lower().replace(' ', '_'), 'notsure', 'no_numbers')):
  # for filedir, dirs, files in os.walk(os.path.join(STORAGE_BASE, REPOSITORY, name.lower().replace(' ', '_'), 'sure', 'numbers')):
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

def rotate_OT(root):
  for filedir, dirs, files in os.walk(root):
    dirs.sort()
    if filedir.split(' ')[-1][0:2] == 'OT':
      files.sort()
      for file in files:
        img = cv2.imread(os.path.join(filedir, file))
        height, width, _ = img.shape
        if width < height:
          img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
          cv2.imwrite(os.path.join(filedir, file), img)
        else:
          break

def copy_OT(root, dest):
  for filedir, dirs, files in os.walk(root):
    dirs.sort()
    if filedir.split(' ')[-1][0:2] == 'OT':
      # print(filedir)
      files.sort()
      dest2 = dest + '/' + filedir.split('/')[-2] + '/' + filedir.split('/')[-1].upper()
      for file in files:
        # print(os.path.join(filedir, file) + '   ' + dest2 + '/' + file)
        shutil.copyfile(os.path.join(filedir, file), dest2 + '/' + file)

def show_OT(root):
  for filedir, dirs, files in os.walk(root):
    dirs.sort()
    if filedir.split(' ')[-1][0:2] == 'OT':
      print(filedir)
# set_GPUs()

def prepare_title_domenica_corriere(root, dest='/home/sunvod/sormani_CNN/giornali/la_domenica_del_corriere/X'):
  for filedir, dirs, files in os.walk(root):
    dirs.sort()
    _, _, files = next(os.walk(dest))
    file_count = len(files) + 1
    for file in files:
      img = cv2.imread(os.path.join(filedir, file), cv2.IMREAD_GRAYSCALE)
      w, h = img.shape
      crop = img[0:1200, 0:w]
      crop = cv2.resize(crop, (224, 224), Image.Resampling.LANCZOS)
      cv2.imwrite(os.path.join(dest, 'img_' + str(file_count)) + '.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
      file_count += 1

def rename_title_domenica_corriere(dest='/home/sunvod/sormani_CNN/giornali/la_domenica_del_corriere/X'):
  _, _, files = next(os.walk(dest))
  file_count = len(files) + 1
  for file in files:
    img = cv2.imread(os.path.join(dest, file))
    cv2.imwrite( os.path.join(dest, 'img_' + str(file_count)) + '.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    file_count += 1

def delete_test_files(root):
  for filedir, dirs, files in os.walk(root):
    for file in files:
      fs = file.split('_')[-1]
      if fs != '1.tif' and fs != '2.tif':
        # print(file)
        os.remove(os.path.join(filedir, file))

def divide_image(root=None):
  if root is None:
    root = os.getcwd()
  for filedir, dirs, files in os.walk(root):
    for file in files:
      file_name = os.path.join(filedir, file)
      file_name_no_ext = Path(file).stem
      file_path_no_ext = os.path.join(filedir, file_name_no_ext)
      ext = Path(file_name).suffix
      if ext != '.tif':
        continue
      img = cv2.imread(file_name)
      h, w, _ = img.shape
      if w < h:
        continue
      imgs = []
      imgs.append(img[0:h, 0 : w // 2])
      imgs.append(img[0:h, w // 2 : w])
      for i, _img in enumerate(imgs):
        cv2.imwrite(file_path_no_ext + '_' + str(i + 1) + ext, _img)
      # os.remove(file_name)


def build_firstpage_csv(root):
  l = []
  for filedir, dirs,files in os.walk(root):
    for file in files:
      date = (file.split('_')[0])
      date = dt.strptime(date, '%Y-%m-%d').date()
      n = file.split('_')[2].split('.')[0]
      l.append([date, n, file])
  l.sort()
  with open(os.path.join(STORAGE_BASE, 'firstpage.csv'), 'w') as f:
    writer = csv.writer(f)
    for e in l:
      writer.writerow(e)

def check_firstpage_csv():
  nl = []
  with open(os.path.join(STORAGE_BASE, 'firstpage.csv'), 'r') as f:
    l = csv.reader(f)
    _n = None
    for i, e in enumerate(l):
      date = dt.strptime(e[0], '%Y-%m-%d').date()
      n = int(e[1])
      file = e[2]
      if _n is None:
        _file = file
        _n = n
        continue
      # if _n + 2 == n:
      #   _file = file
      #   _n = n
      #   continue
      if n - _n != 1:
        nl.append([date, _n, _file])
      _n = n
      _file = file
  nl.sort()
  with open(os.path.join(STORAGE_BASE, 'firstpage_r.csv'), 'w') as f:
    writer = csv.writer(f)
    for e in nl:
      writer.writerow(e)


def renumerate_frames(root):
  for filedir, dirs,files in os.walk(root):
    files.sort()
    for i, file in enumerate(files):
      new_file = 'Scan_' + str(i + 1) + '.tif'
      os.rename(os.path.join(filedir, file), os.path.join(filedir, new_file))


def rename_frames(root):
  for filedir, dirs,files in os.walk(root):
    files.sort()
    for i, file in enumerate(files):
      s = file.split('.')[0]
      date = s.split('_')[0]
      n = ('000000' + s.split('_')[1])[-5:]
      if not n.isdigit():
        n = ('000000' + s.split('_')[2])[-5:]
      new_name = date + '_' + n + '.jpeg'
      os.rename(os.path.join(filedir, file), os.path.join(filedir, new_name))

def convert_crops(root):
  for filedir, dirs,files in os.walk(root):
    files.sort()
    for i, file in enumerate(files):
      img = cv2.imread(os.path.join(filedir, file), cv2.IMREAD_GRAYSCALE)
      ret, thresh = cv2.threshold(img, 48, 255, cv2.THRESH_BINARY)
      cv2.imwrite(os.path.join(filedir, file), thresh, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def reallocate_frame_la_domenica_Del_corriere(csv_file='list_first_pages.csv'):
  with open(os.path.join(STORAGE_BASE, csv_file), 'r') as f:
    csv_file = csv.reader(f)
    csv_list = []
    for row in csv_file:
      csv_list.append(row)
    for i, row in enumerate(csv_list):
      if i < len(csv_list) - 1:
        _row = csv_list[i + 1]
        _n = int(_row[1])
      else:
        _n = n + 100
      filedir = os.path.join(IMAGE_ROOT, IMAGE_PATH, 'La Domenica del Corriere', row[0].split('-')[0], row[0].split('-')[1], row[0].split('-')[2])
      year = row[3].split('/')[2]
      if len(year) == 2 and year == '99':
        year = '1899'
      elif len(year) == 2:
        year = '19' + year
      filedir_destination = os.path.join(IMAGE_ROOT, IMAGE_PATH, '_La Domenica del Corriere', year, ('0' + row[3].split('/')[1])[-2:], ('0' + row[3].split('/')[0])[-2:])
      __n = int(row[1])
      for n in range(__n, _n):
        file_name = 'Scan_' + str(n) + '.tif'
        file_path = os.path.join(filedir, file_name)
        file_path_destination = os.path.join(filedir_destination, file_name)
        if os.path.isfile(file_path):
          print(file_path, file_path_destination)
          Path(filedir_destination).mkdir(parents=True, exist_ok=True)
          shutil.copyfile(file_path, file_path_destination)

def reallocate_frame_il_mondo(csv_file='first_pages_il_mondo.csv'):
  with open(os.path.join(STORAGE_BASE, csv_file), 'r') as f:
    csv_file = csv.reader(f)
    csv_list = []
    for row in csv_file:
      csv_list.append(row)
    for i, row in enumerate(csv_list):
      if i < len(csv_list) - 1:
        _row = csv_list[i + 1]
        _n = int(_row[1])
      else:
        _n = n + 100
      filedir = os.path.join(IMAGE_ROOT, IMAGE_PATH, 'Il Mondo', row[0].split('-')[0], row[0].split('-')[1], row[0].split('-')[2])
      year = row[3].split('/')[2]
      if len(year) == 2:
        year = '19' + year
      filedir_destination = os.path.join(IMAGE_ROOT, IMAGE_PATH, '_Il Mondo', year, ('0' + row[3].split('/')[1])[-2:], ('0' + row[3].split('/')[0])[-2:])
      __n = int(row[1])
      for n in range(__n, _n):
        file_name = 'Scan_' + str(n) + '.tif'
        file_path = os.path.join(filedir, file_name)
        file_path_destination = os.path.join(filedir_destination, file_name)
        if os.path.isfile(file_path):
          print(file_path, file_path_destination)
          Path(filedir_destination).mkdir(parents=True, exist_ok=True)
          shutil.copyfile(file_path, file_path_destination)

def set_folders_pdf_name(root='/mnt/storage01/sormani/JPG-PDF/Il Sole 24 Ore/2016'):
  for filedir, dirs, files in os.walk(root):
    dirs.sort()
    if not len(dirs) or len(dirs) == 3:
      continue
    for dir in dirs:
      if len(dir.split()) <=3:
        continue
      new_dir = ' '.join(dir.split()[:3])
      # print(os.path.join(filedir, dir),' ;' ,  os.path.join(filedir, new_dir))
      os.rename(os.path.join(filedir, dir), os.path.join(filedir, new_dir))

# def invert_left_right_pages(root='/mnt/storage01/sormani/TIFF/Il Mondo/1950/01/01'):
def invert_left_right_pages(root='/mnt/storage01/sormani/TIFF/Il Mondo/1951/01/10'):
  for filedir, dirs, files in os.walk(root):
    if not len(files):
      continue
    files.sort()
    for file in files:
      if file.split('_')[-2] == '':
        continue
      file_name_no_ext = Path(file).stem
      file_path_no_ext = os.path.join(filedir, file_name_no_ext)
      ext = Path(file).suffix
      n = file_name_no_ext.split('_')[-1]
      new_file = '_'.join(file_name_no_ext.split('_')[:-1]) + ('__1' if str(n) == '2' else '__2')
      # print(os.path.join(filedir, file), ' ;', os.path.join(filedir, new_file) + '.tif')
      os.rename(os.path.join(filedir, file), os.path.join(filedir, new_file) + '.tif')

def cover_missing_pages(root='/mnt/storage01/sormani/TIFF/Il Mondo/1950/01/', source='01'):
  root = os.path.join(root, source)
  for filedir, dirs, files in os.walk(root):
    if not len(files):
      continue
    files.sort()
    _n1 = None
    _n2 = None
    _file_name_no_ext = None
    for file in files:
      file_name_no_ext = Path(file).stem
      file_path_no_ext = os.path.join(filedir, file_name_no_ext)
      ext = Path(file).suffix
      n1 = int(file_name_no_ext.split('_')[-2])
      n2 = int(file_name_no_ext.split('_')[-1])
      if _n1 is None:
        _n1 = n1
        _n2 = n2
        _file_name_no_ext = file_name_no_ext
        continue
      if n1 == _n1 + 1 and n2 == _n2:
        if n2 == 1:
          file_missing = '_'.join(_file_name_no_ext.split('_')[:-1]) + '.tif'
        else:
          file_missing = '_'.join(file_name_no_ext.split('_')[:-1]) + '.tif'
        img = cv2.imread(os.path.join('/mnt/storage02/TIFF/Bobine/IL MONDO', source, file_missing), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join('/mnt/storage01/sormani/TIFF/Il Mondo/1960/01/31', file_missing), img)
      _n1 = n1
      _n2 = n2
      _file_name_no_ext = file_name_no_ext

def divide_pdf(root):
  for filedir, dirs, files in os.walk(root):
    count = 1
    files.sort()
    for file in files:
      if file.split('.')[-1] != 'pdf':
        continue
      inputpdf = PdfReader(open(os.path.join(filedir, file), "rb"))
      for i in range(len(inputpdf.pages)):
        output = PdfWriter()
        output.add_page(inputpdf.pages[i])
        new_file = os.path.join(filedir, "Scan_" + ('0000' + str(count))[-4:] + ".pdf")
        with open(new_file, "wb") as outputStream:
          output.write(outputStream)
        pages = convert_from_path(os.path.join(filedir, new_file))
        pages[0].save(os.path.join(filedir, "Scan_" + ('0000' + str(count))[-4:] + ".tif"), "TIFF")
        im = Image.open(os.path.join(filedir, "Scan_" + ('0000' + str(count))[-4:] + ".tif")).convert('L')
        im.save(os.path.join(filedir, "Scan_" + ('0000' + str(count))[-4:] + ".tif"), dpi=(400, 400))
        os.remove(os.path.join(filedir, new_file))
        count += 1
      os.remove(os.path.join(filedir, file))

def rename_files_with_name_folders(root):
  for filedir, dirs, files in os.walk(root):
    files.sort()
    for i, file in enumerate(files):
      parts = filedir.split('/')
      name = parts[-2]
      period = parts[-1].split('-')[1][1:]
      new_file = ('00000' + str(i + 1))[-4:] + ' - ' + name + ' - ' + period
      os.rename(os.path.join(filedir, file), os.path.join(filedir, new_file))



# rename_files_with_name_folders(root='/mnt/storage01/sormani/TIFF/Le Grandi Firme')
# rename_files_with_name_folders(root='/mnt/storage01/sormani/TIFF/La Domenica Del Corriere')
# rename_files_with_name_folders(root='/mnt/storage01/sormani/TIFF/Il Mondo')
# rename_files_with_name_folders(root='/mnt/storage01/sormani/TIFF/Scenario')
# rename_files_with_name_folders(root='/mnt/storage01/sormani/TIFF/Il Secolo Illustrato Della Domenica')

divide_pdf(root='/mnt/storage01/sormani/TIFF/Le Grandi Firme')

# cover_missing_pages(source='29')


# reallocate_frame()

# convert_crops('/home/sunvod/sormani_CNN/X')

# build_firstpage_csv('/home/sunvod/sormani_CNN/firstpage')


# check_firstpage_csv()
#
# rename_frames('/home/sunvod/sormani_CNN/firstpage')

# renumerate_frames('/mnt/storage01/sormani/TIFF/Il Mondo/1950/01')

# delete_test_files('/mnt/storage01/sormani/TIFF/La Domenica del Corriere/1900/01/03')

# prepare_title_domenica_corriere(root='/mnt/storage02/TIFF/La Domenica del Corriere/01/04')

# rename_title_domenica_corriere()
#
#

# count_tiff()

# change_newspaper_name('Osservatore Romano', 'Avvenire', 'Osservatore Romano')

# rename_images_files(ns)

# to_n_classes(ns, n=11, resize=True)

# delete_name('Avvenire')

# get_max_box()

# change_ins_file_name()

# force_to_X()

# set_bobine_images()

# set_bobine_merges()

# rotate_bobine_fotogrammi()

# copy_OT('/mnt/storage02/TIFF/Il Sole 24 ore', '/mnt/storage01/sormani/TIFF/Il Sole 24 Ore/2016')
#
# rotate_OT('/mnt/storage01/sormani/TIFF/Il Sole 24 Ore/2016')
#
# show_OT('/mnt/storage01/sormani/TIFF/Il Sole 24 Ore/2016')x