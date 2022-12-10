from __future__ import annotations

import pathlib
import cv2
import os
import numpy as np

from PyPDF2 import PdfFileMerger
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from PIL import Image
from pathlib import Path
import numpy as np

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import mnist

from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet_v2 import EfficientNetV2M, EfficientNetV2L
from keras.applications.convnext import ConvNeXtXLarge

import warnings
warnings.filterwarnings("ignore")

class Page:
  def __init__(self, file_name, date, newspaper, original_image, pdf_path, jpg_path, txt_path):
    self.original_file_name = file_name
    self.file_name = file_name
    self.original_image = original_image
    self.original_path = str(Path(original_image).parent.resolve())
    self.date = date
    self.year = date.year
    self.month = date.month
    self.month_text = MONTHS[self.month - 1]
    self.day = date.day
    self.newspaper = Newspaper.create(newspaper.name, original_image, newspaper.newspaper_base, date, newspaper.year, newspaper.number)
    self.pdf_path = pdf_path
    self.pdf_file_name = os.path.join(self.pdf_path, 'pdf', self.file_name) + '.pdf'
    self.jpg_path = jpg_path
    self.txt_path = txt_path
    self.txt_file_name = os.path.join(txt_path, self.file_name) + '.txt'
    self.original_txt_file_name = self.txt_file_name
    self.conversions = []
    self.page_control = -1
  def add_conversion(self, conversion):
    if isinstance(conversion, list):
      for conv in conversion:
        self.conversions.append(conv)
    else:
      self.conversions.append(conversion)
  def set_file_names(self):
    page = ('000' + str(self.newspaper.n_page))[-len(str(self.newspaper.n_pages)) : ]
    #page = ('000' + str(self.newspaper.n_page))[-3:]
    self.file_name = self.newspaper.name.replace(' ', '_') \
                     + '_' + str(self.year) \
                     + '_' + str(self.month_text) \
                     + '_' + (str(self.day) if self.day >= 10 else '0' + str(self.day)) \
                     + '_p' + page
    self.txt_file_name = os.path.join(self.txt_path, self.file_name) + '.txt'
  def change_contrast(self):
    if self.force or not self.isAlreadySeen():
      try:
        contrast = self.contrast if self.contrast is not None else self.newspaper.contrast
        image = Image.open(self.original_image)
        pixel_map = image.load()
        if pixel_map[0, 0] == (64, 62, 22) and pixel_map[image.size[0] - 1, image.size[1] - 1] == (64, 62, 22):
          return False
        image = self._change_contrast(image, contrast)
        pixel_map = image.load()
        pixel_map[0, 0] = (64, 62, 22)
        pixel_map[image.size[0] - 1, image.size[1] - 1] = (64, 62, 22)
        image.save(self.original_image)
        return True
      except:
        pass
    return False
  def _change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def save_pages_images(self, storage):
    if self.isAlreadySeen():
      pos = self.newspaper.get_whole_page_location()
      image = Image.open(self.original_image)
      image = image.crop(pos)
      image = image.resize(((int)(image.size[0] * 1.5), (int)(image.size[1] * 1.5)), Image.Resampling.LANCZOS)
      n_files = sum(1 for _, _, files in os.walk(storage) for f in files)
      file_count = str('00000000' + str(n_files))[-7:]
      file_name = os.path.join(storage, file_count + '_' + self.file_name) + pathlib.Path(self.original_image).suffix
      image.save(file_name)
      return True
    return False
  def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)
  def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)
  def _change_contrast_PIL(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def change_contrast_PIL(self, image, level):
    image = self._change_contrast_PIL(image, level)
    return self.convert_from_image_to_cv2(image)
  def change_contrast_cv2(self, img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img
  def cv2_resize(self, img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    return img
  def delete_non_black(self, img):
    min = np.array([20, 20, 20], np.uint8)
    max = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(img, min, max)
    img[mask > 0] = [255, 255, 255]
    return img

  def get_boxes(self, image, level=200, no_resize=False):
    def _get_contours(e):
      return e[0]
    img = self.change_contrast_PIL(image, level)
    img = self.change_contrast_cv2(img)
    np = self.newspaper.get_parameters()
    p = self.file_name.split('_')[-1][1:]
    if np.include is not None:
      if not int(p) in np.include:
        return None, img
    if np.exclude is not None:
      if int(p) in np.exclude:
        return None, img
    img = self.cv2_resize(img, np.scale)
    bimg = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(gray, np.ts, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
      if cv2.contourArea(c) < 100:
        cv2.drawContours(img, [c], -1, (0, 0, 0), -1)
    img = 255 - img
    edges = cv2.Canny(img, 1, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      cv2.rectangle(bimg, (x, y), (x + w, y + h), (36, 255, 12), 2)
    file_name = Path(self.file_name).stem
    images = [(self.file_name + '_00000', bimg)]
    _contours = []
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      _contours.append((x, contour))
    _contours.sort(key = _get_contours)
    for i, (_, contour) in enumerate(_contours):
      x, y, w, h = cv2.boundingRect(contour)
      perimeter = cv2.arcLength(contour, True)
      if hierarchy[0, i, 3] == -1 and perimeter > np.min_perimeter and w > np.box[0] and w < np.box[1] and h > np.box[2] and h < np.box[3]:
          roi = img[y:y + h, x:x + w]
          mean = roi.mean()
          if mean >= np.min_mean and mean <= np.max_mean:
            name = file_name + '_' + ('0000' + str(i + 1))[-5:]
            if not no_resize:
              roi = cv2.resize(roi, NUMBER_IMAGE_SIZE)
            images.append((name, roi, perimeter))
    return images, img
  def get_pages_numbers(self, no_resize = False, filedir = None):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      images, img = self.get_boxes(cropped, no_resize = no_resize)
      if filedir is not None and images is not None:
        for img in images:
          image = Image.fromarray(img[1])
          Path(filedir).mkdir(parents=True, exist_ok=True)
          image.save(os.path.join(filedir, img[0] + '.jpg'), format="JPEG")
        images = None
      return images
    return None
  def get_head(self):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      file_name = Path(self.original_image).stem
      return [[file_name, cropped]]
    return None
  def isAlreadySeen(self):
    l = len(self.newspaper.name)
    f = self.file_name
    return f[: l] == self.newspaper.name.replace(' ', '_') and \
        f[l] == '_' and\
        f[l + 1: l + 5].isdigit() and \
        f[l + 5] == '_' and\
        f[l + 6: l + 9] in MONTHS and\
        f[l + 9] == '_' and\
        f[l + 10: l + 12].isdigit() and\
        f[l+12] == '_' and\
        f[l+13] == 'p' and\
        f[l  + 14: l + 16].isdigit()
  def extract_page(self):
    return self.newspaper.get_page()
  def add_pdf_metadata(self, first_number = None):
    if not os.path.isfile(self.pdf_file_name):
      return
    if first_number is None:
      first_number = 0
    try:
      file_in = open(self.pdf_file_name, 'rb')
      os.rename(self.pdf_file_name, self.pdf_file_name + '.2')
      pdf_merger = PdfFileMerger()
      pdf_merger.append(file_in)
      pdf_merger.addMetadata({
        '/Keywords': 'Nome del periodico:' + self.newspaper.name
                     + ' ; Anno:' + str(self.year)
                     + ' ; Mese:' + str(self.month)
                     + ' ; Giorno:' + str(self.day)
                     + ' ; Numero del quotidiano:' + str(int(self.newspaper.number) + first_number)
                     + ' ; Anno del quotidiano:' + self.newspaper.year,
        '/Title': self.newspaper.name,
        '/Nome_del_periodico': self.newspaper.name,
        '/Anno': str(self.year),
        '/Mese': str(self.month),
        '/Giorno': str(self.day),
        '/Data': str(self.newspaper.date),
        '/Pagina:': str(self.newspaper.n_page),
        '/Numero_del_quotidiano': str(self.newspaper.number),
        '/Anno_del_quotidiano': str(self.newspaper.year),
        '/Producer': 'osi-servizi-informatici.cloud - Milano'
      })
      file_out = open(self.pdf_file_name, 'wb')
      pdf_merger.write(file_out)
      file_in.close()
      file_out.close()
      os.remove(self.pdf_file_name + '.2')
    except:
      os.remove(self.pdf_file_name + '.2')
      file_in.write(self.pdf_file_name)
  def check_pages_numbers(self, model):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      images, img = self.get_boxes(cropped)
      predictions = None
      if images is not None:
        prediction, predictions = self.get_page_numbers(model, images)
      if images is None or prediction is None:
        self.page_control = -1
      elif prediction == self.newspaper.n_page:
        self.page_control = 1
      else:
        self.page_control = 0
      return images, predictions
  def get_page_numbers(self, model, images):
    images.pop(0)
    dataset = []
    for image in images:
      img = cv2.cvtColor(image[1], cv2.COLOR_GRAY2RGB)
      img = Image.fromarray(img)
      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      dataset.append(img)
    try:
      original_predictions = list(np.argmax(model.predict(np.array(dataset), verbose = 0), axis=-1))
    except:
      return None, None
    b = None
    predictions = []
    for e in original_predictions:
      if b is None:
        b = e
        if e == 0:
          continue
      elif e ==  0 and (b == 0 or (b == 10 and  len(predictions) == 0) or len(predictions) >= 2):
        continue
      if e != 10:
        predictions.append(str(e))
      b = e
    predictions = ''.join(predictions)
    if len(predictions):
      prediction = int(predictions)
    else:
      prediction = None
    return prediction, original_predictions

