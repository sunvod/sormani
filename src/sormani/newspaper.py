
from __future__ import annotations

import cv2
import numpy as np
import os
import datetime
import re
from PIL import Image, ImageOps
import pytesseract
from pathlib import Path

from src.sormani.system import MONTHS, exec_ocrmypdf, CONTRAST, STORAGE_DL

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
               exclude_colors = None):
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
    self.exclude_colors = exclude_colors

class Newspaper():
  @staticmethod
  def create(name, file_path, newspaper_base = None, date = None, year = None, number = None):
    if date is None:
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
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
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
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    return parameters
  def __init__(self, newspaper_base, name, file_path, date, year, number, init_page):
    self.newspaper_base = newspaper_base
    self.name = name
    self.file_path = file_path
    self.date = date
    self.contrast = CONTRAST
    if year is not None:
      self.year = year
      self.number = number
    else:
      self.year, self.number = self.get_head()
    self.page = None
    self.init_page = init_page
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
  def get_grayscale(self, image):
      return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  def canny(self, image):
    return cv2.Canny(image, 100, 200)
  def dilate(self, image, l = 5):
    kernel = np.ones((l, l), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
  def erode(self, image, l = 5):
    kernel = np.ones((l, l), np.uint8)
    return cv2.erode(image, kernel, iterations=1)
  def remove_noise(self, image, l = 5):
    return cv2.medianBlur(image, l)
  def thresholding(self, image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  def opening(self, image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  def remove_lines(self, image):
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)
    return cv2.add(temp3, image)
  def crop_png(self, image):
    image = image.crop(self.get_whole_page_location(image))
    return image
  def crop(self, left, top, right, bottom, resize = None, contrast = None, erode = 2, dpi = 600, oem = 3, count = 0, remove_lines = False, old_y = None):
    image = Image.open(self.file_path)
    image = image.crop((left, top, right, bottom))
    if resize is not None:
      image = image.resize((int(image.size[0] * resize), int(image.size[1] * resize)), Image.Resampling.LANCZOS)
    if contrast is not None:
      image = self.change_contrast(image, contrast)
    image.save('temp.tif')
    image = cv2.imread('temp.tif')
    image = self.erode(image, l = erode)
    if remove_lines:
      image = self.remove_lines(image)
    custom_config = r'-l ita --oem ' + str(oem) + ' --psm 4 --dpi ' + str(dpi)
    x = pytesseract.image_to_string(image, config = custom_config)
    x = x.replace("\n", "").strip()
    x = re.sub('[^0-9]+', ' ', x)
    ls = x.split(' ')
    ls = [ r for r in ls if r.isdigit()]
    y = [r for r in ls if len(r)]
    repeat = False
    if len(y) > 1:
      by = [r for r in y if len(r) == 2]
      repeat = not len(by)
      y = by
    elif len(ls) == 1 and len(ls[0]) > 2:
      repeat = True
    if repeat:
      count += 1
      if count < 5:
        if resize is None:
          resize = 1
        resize -= 0.1
        return self.crop(left, top, right, bottom, resize = resize, contrast = contrast, erode = erode, dpi = dpi, count = count)
      else:
        if len(y) > 0 and len(y[0]) == 3 and y[0][0] == '1':
          y = [y[0][1:]]
        else:
          y = []
    if len(y) == 1 and int(y[0]) >= 12 and int(y[0]) <= 20:
      count += 1
      if count < 2:
        return self.crop(left, top, right, bottom, resize=resize, contrast=contrast, erode=erode, dpi=dpi, count=count, remove_lines = True, old_y = y)
      elif old_y is not None:
        y = old_y
    cv2.imwrite('temp.tif', image)
    dimension = os.path.getsize('temp.tif')
    image = Image.open('temp.tif')
    #os.remove('temp.tif')
    return y, image, dimension
  def get_number(self):
    folder_count = 0
    for month in range(self.date.month):
      m = str(month + 1)
      input_path = os.path.join(self.newspaper_base, str(self.date.year), m if len(m) == 2 else '0' + m)
      if os.path.exists(input_path):
        listdir = os.listdir(input_path)
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
    self.init_year = 150
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Giornale', file_path, date, year, number, init_page = 3)
  def get_whole_page_location(self, image):
    whole = (0, 100, 5000, 500)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 40,
                                max_w = 150,
                                min_h = 120,
                                max_h = 240,
                                ts = 220,
                                min_mean = 50,
                                max_mean = 250,
                                fill_hole=3,
                                max_fillarea = 0)

class Il_manifesto(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 46
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Manifesto', file_path, date, year, number, init_page = 3)
  def get_whole_page_location(self, image):
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
    whole = [0, 100, 4850, 400]
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
    whole = [0, 100, 4850, 400]
    return whole

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
    whole = [0, 100, 4850, 400]
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
    whole = [0, 100, 4850, 520]
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
    whole = [0, 100, 4850, 400]
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
    whole = (0, 100, w, 800)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale = 200,
                                min_w = 91 - 50,
                                max_w = 206 + 50,
                                min_h = 235 - 50,
                                max_h = 321 + 50,
                                ts = 170,
                                min_mean = 183.9 - 50,
                                max_mean = 220.4 + 50)

class Il_Foglio(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 21
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Foglio', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 800)
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=10,
                                max_w=500,
                                min_h=10,
                                max_h=500,
                                ts=10,
                                min_mean=0,
                                max_mean=1000)

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
    whole = [0, 100, 4850, 400]
    return whole