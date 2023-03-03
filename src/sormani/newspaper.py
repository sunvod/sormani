
from __future__ import annotations

import cv2
import numpy as np
import os
import datetime
import re
from PIL import Image, ImageOps
import pytesseract
from pathlib import Path
import tensorflow as tf

from src.sormani.system import MONTHS, exec_ocrmypdf, CONTRAST, STORAGE_DL, STORAGE_BASE

import warnings
warnings.filterwarnings("ignore")

class Newspaper_parameters():
  def __init__(self,
               scale,
               min_w,
               max_w,
               min_h,
               max_h,
               ts,
               min_mean = None,
               max_mean = None,
               include = None,
               exclude = [1],
               max_fillarea = 100,
               invert = False,
               internal_box = None,
               fill_hole = None,
               invert_fill_hole=False,
               exclude_colors = None,
               can_be_internal=False,
               max_distance = None,
               left_free=None,
               right_free=None,
               delete_horizontal = False,
               delete_vertical = False,
               min_area=None,
               position='top'):
    self.scale = scale
    self.box = (min_w, max_w, min_h, max_h)
    self.ts = ts
    self.min_mean = min_mean
    self.max_mean = max_mean
    self.include = include
    self.exclude = exclude
    self.max_fillarea = max_fillarea
    self.invert = invert
    self.internal_box = internal_box
    self.fill_hole = fill_hole
    self.invert_fill_hole = invert_fill_hole
    self.exclude_colors = exclude_colors
    self.can_be_internal = can_be_internal
    self.max_distance = max_distance
    self.model = None
    self.left_free = left_free
    self.right_free = right_free
    self.delete_horizontal = delete_horizontal
    self.delete_vertical = delete_vertical
    self.min_area = min_area
    self.position = position

class Newspaper_crop_parameters():
  def __init__(self,
               left,
               right,
               top,
               bottom):
    self.left = left
    self.right = right
    self.top = top
    self.bottom = bottom
class Newspaper():
  @staticmethod
  def create(name, file_path, newspaper_base = None, date = None, year = None, month = None, number = None, model=None):
    if date is None and month is None:
      file_name = Path(file_path).stem
      year = ''.join(filter(str.isdigit, file_name.split('_')[-4]))
      month = MONTHS.index(file_name.split('_')[-3]) + 1
      day = file_path.split('_')[-2]
      if year.isdigit() and day.isdigit():
        date = datetime.date(int(year), int(month), int(day))
      else:
        raise NotADirectoryError('Le directory non indicano una data.')
    if name == 'La Stampa':
      newspaper = La_stampa(newspaper_base, file_path, date, year, number)
    elif name == 'Il Giornale':
      newspaper = Il_Giornale(newspaper_base, file_path, date, year, number)
    elif name == 'Il Manifesto':
      newspaper = Il_manifesto(newspaper_base, file_path, date, year, number)
    elif name == 'Avvenire':
      newspaper = Avvenire(newspaper_base, file_path, date, year, number)
    elif name == 'Milano Finanza':
      newspaper = Milano_Finanza(newspaper_base, file_path, date, year, number)
    elif name == 'Il Fatto Quotidiano':
      newspaper = Il_Fatto_Quotidiano(newspaper_base, file_path, date, year, number)
    elif name == 'Italia Oggi':
      newspaper = Italia_Oggi(newspaper_base, file_path, date, year, number)
    elif name == 'Libero':
      newspaper = Libero(newspaper_base, file_path, date, year, number)
    elif name == 'Alias':
      newspaper = Alias(newspaper_base, file_path, date, year, number)
    elif name == 'Alias Domenica':
      newspaper = Alias_Domenica(newspaper_base, file_path, date, year, number)
    elif name == 'Osservatore Romano':
      newspaper = Osservatore_Romano(newspaper_base, file_path, date, year, number)
    elif name == 'Il Foglio':
      newspaper = Il_Foglio(newspaper_base, file_path, date, year, number)
    elif name == 'Unita':
      newspaper = Unita(newspaper_base, file_path, date, year, number)
    elif name == 'Tutto Libri':
      newspaper = Tutto_Libri(newspaper_base, file_path, date, year, number)
    elif name == 'Il Giorno':
      newspaper = Il_Giorno(newspaper_base, file_path, date, year, number)
    elif name == 'La Gazzetta dello Sport':
      newspaper = La_Gazzetta(newspaper_base, file_path, date, year, number)
    elif name == 'Scenario':
      newspaper = Scenario(newspaper_base, file_path, date, year, number)
    elif name == 'La Domenica del Corriere':
      newspaper = La_Domenica_del_Corriere(newspaper_base, file_path, date, year, number)
    elif name == 'Il Mondo':
      newspaper = Il_Mondo(newspaper_base, file_path, date, year, number)
    elif name == 'Il Sole 24 Ore':
      newspaper = Il_Sole_24_Ore(newspaper_base, file_path, date, year, number)
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    newspaper.month = month
    newspaper.model = model
    newspaper.n_page = None
    return newspaper

  @staticmethod
  def get_parameters(name):
    if name == 'La Stampa':
      parameters = La_stampa.get_parameters()
    elif name == 'Il Giornale':
      parameters = Il_Giornale.get_parameters()
    elif name == 'Il Manifesto':
      parameters = Il_manifesto.get_parameters()
    elif name == 'Avvenire':
      parameters = Avvenire.get_parameters()
    elif name == 'Milano Finanza':
      parameters = Milano_Finanza.get_parameters()
    elif name == 'Il Fatto Quotidiano':
      parameters = Il_Fatto_Quotidiano.get_parameters()
    elif name == 'Italia Oggi':
      parameters = Italia_Oggi.get_parameters()
    elif name == 'Libero':
      parameters = Libero.get_parameters()
    elif name == 'Alias':
      parameters = Alias.get_parameters()
    elif name == 'Alias Domenica':
      parameters = Alias_Domenica.get_parameters()
    elif name == 'Osservatore Romano':
      parameters = Osservatore_Romano.get_parameters()
    elif name == 'Il Foglio':
      parameters = Il_Foglio.get_parameters()
    elif name == 'Unita':
      parameters = Unita.get_parameters()
    elif name == 'Tutto Libri':
      parameters = Tutto_Libri.get_parameters()
    elif name == 'Il Giorno':
      parameters = Il_Giorno.get_parameters()
    elif name == 'La Gazzetta dello Sport':
      parameters = La_Gazzetta.get_parameters()
    elif name == 'Il Sole 24 Ore':
      parameters = Il_Sole_24_Ore.get_parameters()
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    return parameters
  def get_ins_parameters(self):
    return self.get_parameters()
  def __init__(self, newspaper_base, name, file_path, date, year, number, init_page, model=None):
    self.newspaper_base = newspaper_base
    self.name = name
    self.file_path = file_path
    self.date = date
    self.contrast = CONTRAST
    if year is not None:
      self.year = year
      if number is None:
        _, number = self.get_head()
      self.number = number
    else:
      self.year, self.number = self.get_head()
    self.page = None
    self.init_page = init_page
    self.model = model
  def check_n_page(self, date):
    file_name = Path(self.file_path).stem
    l = len(self.name)
    year = file_name[l + 1 : l + 5]
    month = file_name[l + 6 : l + 9]
    day = file_name[l + 10 : l + 12]
    if month in MONTHS:
      month = str(MONTHS.index(month) + 1)
    if year.isdigit() and month.isdigit() and day.isdigit():
      file_date = datetime.date(int(year), int(month), int(day))
      return date == file_date
    return False
  def change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def crop_png(self, image):
    dims = self.get_whole_page_location(image)
    if isinstance(dims[0], list):
      # img = None
      imgs = []
      for dim in dims:
      #   _img = np.asarray(image.crop(dim))
      #   if img is None:
      #     img = _img
      #   else:
      #     img = np.concatenate((img, _img), axis=1)
      # image = Image.fromarray(img)
        imgs.append(image.crop(dim))
    else:
      imgs = image.crop(dims)
    return imgs
  def crop_ins_png(self, image):
    dims = self.get_ins_whole_page_location(image)
    if isinstance(dims[0], list):
      # img = None
      imgs = []
      for dim in dims:
      #   _img = np.asarray(image.crop(dim))
      #   if img is None:
      #     img = _img
      #   else:
      #     img = np.concatenate((img, _img), axis=1)
      # image = Image.fromarray(img)
        imgs.append(image.crop(dim))
    else:
      imgs = image.crop(dims)
    return imgs
  def get_ins_whole_page_location(self, image):
    return self.get_whole_page_location(image)
  def get_number(self):
    folder_count = 0
    for month in range(self.date.month):
      m = str(month + 1)
      input_path = os.path.join(self.newspaper_base, str(self.date.year), m if len(m) == 2 else '0' + m)
      if os.path.exists(input_path):
        listdir = os.listdir(input_path)
        listdir.sort()
        listdir = [x for x in listdir if x.isdigit()]
        listdir.sort(key=self._get_number_sort)
        for folders in listdir:
          if month + 1 == self.date.month and int(folders) > self.date.day:
            return str(folder_count)
          if os.path.isdir(os.path.join(input_path, folders)):
            folder_count += 1
    return str(folder_count)
  def _get_number_sort(self, e):
    if e.isdigit():
      return int(e)
    return 0
  def get_head(self):
    number = self.get_number()
    year = self.init_year + self.date.year - 2016
    if self.year_change is not None and \
        (self.year_change[1] < self.date.month or \
        (self.year_change[1] == self.date.month and self.year_change[0] <= self.date.day)):
      year += 1
    return str(year), number
  def get_page(self):
    return None, None
  def set_n_pages(self, page_pool, n_pages):
    f = 1
    l = n_pages
    r = 2
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      n = Path(page.file_name).stem.split('_')[-1]
      if n != '0':
        count_zero = 0
        if r == 2:
          page.newspaper.n_page = f
          f += 1
          r = -1
        elif r == -1:
          page.newspaper.n_page = l
          l -= 1
          r = -2
        elif r == -2:
          page.newspaper.n_page = l
          l -= 1
          r = 1
        elif r == 1:
          page.newspaper.n_page = f
          f += 1
          r = 2
      else:
        if count_zero < 2:
          page.newspaper.n_page = f
          f += 1
        else:
          page.newspaper.n_page = l
          l -= 1
        r = 2
        count_zero += 1
  def get_dictionary(self):
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']
  def get_prefix(self):
    return ''
  def get_crop_parameters(self, i, width, height):
    if i == 0:
      left = width // 2 + 1
      top = 0
      right = width
      bottom = height
    elif i == 1:
      left = 0
      top = 0
      right = width // 2
      bottom = height
    else:
      left = None
      top = None
      right = None
      bottom = None
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def get_remove_borders_parameters(self, i, width, height):
    left = 0
    top = 0
    right = width
    bottom = height
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def divide(self, img):
    imgs = []
    height, width, _ = img.shape
    parameters = self.get_crop_parameters(0, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    parameters = self.get_crop_parameters(1, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    if os.path.dirname(self.file_path).split(' ')[-1][0:2] == 'OT':
      try:
        s = os.path.dirname(self.file_path).split(' ')[-1][2:]
        f = int(s.split('-')[0])
      except Exception as e:
        error = 'Folder ' + self.filedir + ' ' + 'is not valid'
        raise(error)
      if f == 4:
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        height, width, _ = img1.shape
        imgs.append(img1[0:height, 0:width // 2])
        imgs.append(img1[0:height, width // 2:width])
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width, _ = img2.shape
        imgs.append(img2[0:height, 0:width // 2])
        imgs.append(img2[0:height, width // 2:width])
        # imgs.append(img1)
        # imgs.append(img2)
    if not len(imgs):
      imgs.append(img1)
      imgs.append(img2)
    return imgs

  # image1 = image.crop((parameters.left, parameters.top, parameters.right, parameters.bottom))
  def is_first_page(self, model):
    return False
  def get_ofset(self):
    return 0, -200, 0, 0
  def get_dimension(self, img):
    _x, _y, _w, _h = cv2.boundingRect(img)
    return w, h

class La_stampa(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 150
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Stampa', file_path, date, year, number, init_page = 3)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 500)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 39 - 20,
                                max_w = 91 + 20,
                                min_h = 125 - 20,
                                max_h = 146 + 20,
                                ts = 170,
                                min_mean = 146.2 - 50,
                                max_mean = 191.0 + 50)

class Il_Giornale(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = (43, 36)
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Giornale', file_path, date, year, number, init_page = 3)
  def get_number(self):
    folder_count = 0
    for month in range(self.date.month):
      m = str(month + 1)
      input_path = os.path.join(self.newspaper_base, str(self.date.year), m if len(m) == 2 else '0' + m)
      if os.path.exists(input_path):
        listdir = os.listdir(input_path)
        listdir.sort()
        listdir = [x for x in listdir if x.isdigit()]
        listdir.sort(key=self._get_number_sort)
        for folders in listdir:
          if month + 1 == self.date.month and int(folders) > self.date.day:
            return str(folder_count)
          if os.path.isdir(os.path.join(input_path, folders)):
            day = int(folders)
            if self.date.weekday() != 0:
              try:
                if datetime.datetime(self.date.year, month + 1 , day).weekday() != 0:
                  folder_count += 1
              except Exception as e:
                pass
            else:
              if datetime.datetime(self.date.year, month + 1, day).weekday() == 0:
                folder_count += 1
    return str(folder_count)
  def get_head(self):
    number = self.get_number()
    if self.date.weekday() != 0:
      year = self.init_year[0] + self.date.year - 2016
    else:
      year = self.init_year[1] + self.date.year - 2016
    return str(year), number
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 500)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 75,
                                max_w = 150,
                                min_h = 120,
                                max_h = 240,
                                ts = 220,
                                min_mean = 50,
                                fill_hole=3,
                                # invert_fill_hole=False,
                                max_mean = 250)

class Il_manifesto(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 46
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Manifesto', file_path, date, year, number, init_page = 3)
  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page % 2 == 0:
      whole = [0, 150, 1000, 450]
    else:
      whole = [3850, 150, 4850, 450]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=30,
                                max_w=60,
                                min_h=55,
                                max_h=75,
                                ts=5,
                                min_mean=20,
                                max_mean=140)

class Milano_Finanza(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 27
    self.year_change = [1, 9]
    Newspaper.__init__(self, newspaper_base, 'Milano Finanza', file_path, date, year, number, init_page = 5)
  def set_n_pages(self, page_pool, n_pages):
    f = 1
    l = n_pages
    r = 2
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      n = Path(page.file_name).stem.split('_')[-1]
      if n != '0':
        if r == 2:
          page.newspaper.n_page = f
          f += 1
          r = -1
        elif r == -1:
          page.newspaper.n_page = l
          l -= 1
          if n_page > 2:
            r = -2
          else:
            r = 2
        elif r == -2:
          page.newspaper.n_page = l
          l -= 1
          r = 1
        elif r == 1:
          page.newspaper.n_page = f
          f += 1
          r = 2
      else:
        page.newspaper.n_page = f
        f += 1
        r = -2
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [0, 100, w, 400]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 30,
                                max_w = 100,
                                min_h = 90,
                                max_h = 200,
                                ts = 170,
                                min_mean = 10,
                                max_mean = 500)

class Il_Fatto_Quotidiano(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 8
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Fatto Quotidiano', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [100, 100, w, 500]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=40,
                                max_w=130,
                                min_h=100,
                                max_h=170,
                                ts=150,
                                min_mean=140,
                                max_mean=250)

class Italia_Oggi(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 25
    self.year_change = [8, 8]
    Newspaper.__init__(self, newspaper_base, 'Italia Oggi', file_path, date, year, number, init_page = 5)
  def set_n_pages(self, page_pool, n_pages):
    f = 1
    l = n_pages
    r = 2
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      n = Path(page.file_name).stem.split('_')[-1]
      if n != '0':
        if r == 2:
          page.newspaper.n_page = f
          f += 1
          r = -1
        elif r == -1:
          page.newspaper.n_page = l
          l -= 1
          r = -2
        elif r == -2:
          page.newspaper.n_page = l
          l -= 1
          r = 1
        elif r == 1:
          page.newspaper.n_page = f
          f += 1
          r = 2
      else:
        page.newspaper.n_page = f
        f += 1
        r = 2
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [0, 100, w, 400]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=70,
                                max_w=110,
                                min_h=110,
                                max_h=180,
                                ts=10,
                                min_mean=100,
                                max_mean=200)
class Libero(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 51
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Libero', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [0, 100, w, 520]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=50,
                                max_w=130,
                                min_h=110,
                                max_h=190,
                                ts=1,
                                min_mean=40,
                                max_mean=160,
                                fill_hole = 2,
                                exclude_colors = (-230, 20, 20),
                                invert = True)

class Alias(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 19
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Alias', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [0, 100, w, 400]
    return whole

class Alias_Domenica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 6
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Alias Domenica', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole

class Avvenire(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 49
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Avvenire', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 800)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 80,
                                max_w = 220,
                                min_h = 220,
                                max_h = 330,
                                ts = 170,
                                min_mean = 120,
                                max_mean = 260)
class Osservatore_Romano(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 156
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Osservatore Romano', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page % 2 == 0:
      whole = [0, 100, 1000, 800]
    else:
      whole = [w - 1000, 150, w, 800]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=35,
                                max_w=70,
                                min_h=45,
                                max_h=130,
                                ts=250,
                                min_mean=100,
                                max_mean=200)

class Il_Foglio(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 21
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Foglio', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 300, w / 2, 620)
    return whole
  def get_dictionary(self):
    return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'G', 'P', 'X']
  def get_prefix(self):
    return 'PAG'
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=50,
                                max_w=100,
                                min_h=50,
                                max_h=150,
                                ts=10,
                                min_mean=10,
                                max_mean=500)

class Unita(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 93
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Unita', file_path, date, year, number, init_page=5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, h - 700, w, h - 300)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=180,
                                max_w=240,
                                min_h=180,
                                max_h=240,
                                ts=100,
                                min_mean=0,
                                max_mean=500,
                                invert = True,
                                internal_box = (30, 100, 80, 120))

class Tutto_Libri(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 19
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Tutto Libri', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = [0, 100, w, 400]
    return whole

class Il_Giorno(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Giorno', file_path, date, year, number, init_page = 3)
  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page % 2 == 0:
      whole = [0, 150, 1000, 500]
    else:
      whole = [w - 1000, 150, w, 500]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 10,
                                max_w = 120,
                                min_h = 90,
                                max_h = 150,
                                ts = 240,
                                min_mean = 0,
                                max_mean = 500,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)
class Il_Sole_24_Ore(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 152
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Sole 24 Ore', file_path, date, year, number, init_page = 3)
    self.contrast = 10
  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page is None:
      whole = [[200, 200, 600, 650], [w - 450, 200, w - 150, 650]]
    elif self.n_page % 2 == 0:
      whole = [[200, 200, 500, 650], [w - 450, 200, w - 150, 650]]
    else:
      whole = [[200, 200, 500, 650], [w - 450, 200, w - 150, 650]]
    return whole
  def get_ins_whole_page_location(self, image):
    w, h = image.size
    if self.n_page is None:
      whole = [[200, 200, 470, 620], [w - 450, 200, w - 150, 650], [200, h - 550, 620, h - 200],
               [w // 2 - 250, h - 600, w // 2 + 250, h - 200], [w - 620, h - 550, w - 200, h - 200]]
    elif self.n_page % 2 == 0:
      whole = [[200, 200, 470, 620], [w - 450, 200, w - 150, 650], [200, h - 550, 620, h - 200], [w // 2 - 250, h - 600, w // 2 + 250, h - 200]]
    else:
      whole = [[200, 200, 470, 620], [w - 470, 200, w - 200, 620], [w - 620, h - 550, w - 200, h - 200], [w // 2 - 250, h - 600, w // 2 + 250, h - 200]]
    return whole
  def set_n_pages(self, page_pool, n_pages):
    f = 1
    m = None
    if page_pool.filedir.split(' ')[-1][0] == 'P':
      try:
        f = int(page_pool.filedir.split(' ')[-1][1:])
        n_pages += f - 1
      except Exception as e:
        error = 'Folder ' + page_pool.filedir + ' ' + 'is not valid'
        raise(error)
    elif page_pool.filedir.split(' ')[-1][0:2] == 'OT':
      m = [x for x in range(n_pages)]
      if n_pages == 8:
        m = [8, 1, 4, 5, 6, 3, 2, 7]
      elif n_pages == 16:
        m = [16, 1, 8, 9, 10, 7, 2, 15, 14, 3, 6, 11, 12, 5, 4, 13]
      elif n_pages == 32:
        # m = [64,81,80,65,82,63,66,79,62,83,78,67,84,61,68,77,60,85,76,69,86,59,70,75,58,87,74,71,88,57,72,73]
        m = [8, 25, 24, 9, 26, 7, 10, 23, 6, 27, 22, 11, 28, 5, 12, 21, 4, 29, 20, 13, 30, 3, 14, 19, 2, 31, 18, 15, 32, 1, 16, 17]
      elif n_pages == 40:
        # m = [56,17,36,37,38,35,18,55,54,19,34,39,40,33,20,53,52,21,32,41,42,31,22,51,50,23,30,43,44,29,24,49,48,25,28,45,46,27,26,47]
        m = [40, 1, 20, 21, 22, 19, 2, 39, 38, 3, 18, 23, 24, 17, 4, 37, 36, 5, 16, 25, 26, 15, 6, 35, 34, 7, 14, 27, 28, 13, 8, 33, 32, 9, 12, 29, 30, 11, 10, 31]
      if len(page_pool.filedir.split(' ')[-1].split('-')) > 1:
        n = int(page_pool.filedir.split(' ')[-1].split('-')[-1])
        try:
          m = list(np.array(m) + n - 1)
        except:
          pass
    l = n_pages
    r = 2
    lasffl = None
    for n_page, page in enumerate(page_pool):
      if page.newspaper.n_page is not None:
        continue
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      if m is not None:
        page.newspaper.n_page = m[n_page]
        continue
      n = Path(page.file_name).stem.split('_')[-1]
      if n != '0':
        if r == 2:
          page.newspaper.n_page = f
          f += 1
          r = -1
        elif r == -1:
          page.newspaper.n_page = l
          l -= 1
          r = -2
        elif r == -2:
          page.newspaper.n_page = l
          l -= 1
          r = 1
        elif r == 1:
          page.newspaper.n_page = f
          f += 1
          r = 2
      else:
        if page.prediction is not None:
          if page.prediction == f:
            page.newspaper.n_page = f
            f += 1
            lasffl = 'f'
          elif page.prediction == l:
            page.newspaper.n_page = l
            l -= 1
            lasffl = 'l'
          elif str(page.prediction) in str(f):
            page.newspaper.n_page = f
            f += 1
            lasffl = 'f'
          elif str(page.prediction) in str(l):
            page.newspaper.n_page = l
            l -= 1
            lasffl = 'l'
          else:
            page.newspaper.n_page = page.prediction
        else:
          if lasffl == 'f':
            page.newspaper.n_page = f
            f += 1
          else:
            page.newspaper.n_page = l
            l -= 1
        r = 2
  def get_ins_parameters(self):
    return [Newspaper_parameters(scale=200,
                                 min_w=30,
                                 max_w=120,
                                 min_h=45,
                                 max_h=160,
                                 ts=240,
                                 min_mean=0,
                                 max_mean=500,
                                 invert=False,
                                 fill_hole=1,
                                 invert_fill_hole=True,
                                 max_distance=10,
                                 can_be_internal=True,
                                 left_free=(100, 36),
                                 right_free = (100, 36),
                                 # delete_horizontal=True,
                                 # delete_vertical=True,
                                 min_area=500),
            Newspaper_parameters(scale=200,
                                 min_w=30,
                                 max_w=120,
                                 min_h=45,
                                 max_h=160,
                                 ts=240,
                                 min_mean=0,
                                 max_mean=500,
                                 invert=False,
                                 fill_hole=1,
                                 invert_fill_hole=True,
                                 max_distance=10,
                                 can_be_internal=True,
                                 left_free=(100, 36),
                                 right_free=(100, 36),
                                 delete_horizontal=True,
                                 # delete_vertical=True,
                                 min_area=500),
            Newspaper_parameters(scale=200,
                                 min_w=30,
                                 max_w=120,
                                 min_h=45,
                                 max_h=160,
                                 ts=240,
                                 min_mean=0,
                                 max_mean=500,
                                 invert=False,
                                 fill_hole=1,
                                 invert_fill_hole=True,
                                 max_distance=10,
                                 can_be_internal=True,
                                 left_free=(100, 36),
                                 right_free=(100, 36),
                                 delete_horizontal=True,
                                 # delete_vertical=True,
                                 min_area=500,
                                 position='bottom'),
            Newspaper_parameters(scale=200,
                                 min_w=30,
                                 max_w=120,
                                 min_h=90,
                                 max_h=160,
                                 ts=240,
                                 min_mean=0,
                                 max_mean=500,
                                 invert=False,
                                 fill_hole=1,
                                 invert_fill_hole=True,
                                 max_distance=10,
                                 can_be_internal=True,
                                 left_free=(100, 36),
                                 right_free=(100, 36),
                                 delete_horizontal=True,
                                 # delete_vertical=True,
                                 min_area=500,
                                 position='bottom')]
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 30,
                                max_w = 120,
                                min_h = 45,
                                max_h = 160,
                                ts = 240,
                                min_mean = 60,
                                max_mean = 220,
                                invert=False,
                                fill_hole=1,
                                invert_fill_hole=True,
                                max_distance=10,
                                can_be_internal=True,
                                left_free=(100, 36),
                                right_free=(100, 36),
                                delete_horizontal=True,
                                # delete_vertical=True,
                                min_area=500)
class La_Gazzetta(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year =  (120, 72)
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Gazzetta dello Sport', file_path, date, year, number, init_page = 3)
    self.contrast = None
  def get_number(self):
    folder_count = 0
    for month in range(self.date.month):
      m = str(month + 1)
      input_path = os.path.join(self.newspaper_base, str(self.date.year), m if len(m) == 2 else '0' + m)
      if os.path.exists(input_path):
        listdir = os.listdir(input_path)
        listdir.sort()
        listdir = [x for x in listdir if x.isdigit()]
        listdir.sort(key=self._get_number_sort)
        for folders in listdir:
          if month + 1 == self.date.month and int(folders) > self.date.day:
            return str(folder_count)
          if os.path.isdir(os.path.join(input_path, folders)):
            day = int(folders)
            if self.date.weekday() != 6:
              try:
                if datetime.datetime(self.date.year, month + 1 , day).weekday() != 6:
                  folder_count += 1
              except Exception as e:
                pass
            else:
              if datetime.datetime(self.date.year, month + 1, day).weekday() == 6:
                folder_count += 1
    return str(folder_count)
  def get_head(self):
    number = self.get_number()
    if self.date.weekday() != 6:
      year = self.init_year[0] + self.date.year - 2016
    else:
      year = self.init_year[1] + self.date.year - 2016
    return str(year), number
  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page % 2 == 0:
      whole = [0, 150, 1000, 500]
    else:
      whole = [w - 1000, 150, w, 500]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 10,
                                max_w = 120,
                                min_h = 90,
                                max_h = 150,
                                ts = 240,
                                min_mean = 0,
                                max_mean = 500,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)
class Scenario(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Scenario', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page

  def get_remove_borders_parameters(self, i, width, height):
    o = 400
    if i == 0:
      left = o * 2
      top = o
      right = width
      bottom = height - o
    elif i == 1:
      left = 0
      top = o
      right = width - (o * 2)
      bottom = height - o
    else:
      left = o
      top = o
      right = width - o
      bottom = height - o
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 10,
                                max_w = 120,
                                min_h = 90,
                                max_h = 150,
                                ts = 240,
                                min_mean = 0,
                                max_mean = 500,
                                # fill_hole=4,
                                # invert_fill_hole=True,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)

class La_Domenica_del_Corriere(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Domenica del Corriere', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = ((w + 100) // 2, 200 + 800, w - 800, 1700)
    return whole
  def set_n_pages(self, page_pool, n_pages):
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page

  def get_remove_borders_parameters(self, i, width, height):
    left = 950
    top = 800
    right = width - 800
    bottom = height - 900
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def is_first_page(self, model):
    if model is None:
      return False
    dataset = []
    image = Image.open(self.original_image)
    cropped = self.newspaper.crop_png(image)
    img = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    dataset.append(img)
    try:
      original_predictions = list(np.argmax(self.model.predict(np.array(dataset), verbose=0), axis=-1))
    except Exception as e:
      return None, None, None
    pass
  def get_ofset(self):
    return 0, -200, 0, 200
  def get_dimension(self, img=None):
    return 5600, 7400
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 10,
                                max_w = 120,
                                min_h = 90,
                                max_h = 150,
                                ts = 240,
                                min_mean = 0,
                                max_mean = 500,
                                # fill_hole=4,
                                # invert_fill_hole=True,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)

class Il_Mondo(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Mondo', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_remove_borders_parameters(self, i, width, height):
    o = 200
    if i == 0:
      left = o * 2
      top = o
      right = width
      bottom = height - o
    elif i == 1:
      left = 0
      top = o
      right = width - (o * 2)
      bottom = height - o
    else:
      left = None
      top = None
      right = None
      bottom = None
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 10,
                                max_w = 120,
                                min_h = 90,
                                max_h = 150,
                                ts = 240,
                                min_mean = 0,
                                max_mean = 500,
                                # fill_hole=4,
                                # invert_fill_hole=True,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)
