
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
from system import *

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
               less_w=20,
               less_h=20,
               min_area=None,
               max_near=2,
               n_digits=3):
    m = scale / 200
    self.scale = scale
    self.box = (int(min_w * m), int(max_w * m), int(min_h * m), int(max_h * m))
    self.ts = ts
    self.min_mean = min_mean
    self.max_mean = max_mean
    self.include = include
    self.exclude = exclude
    self.max_fillarea = int(max_fillarea * m)
    self.invert = invert
    self.internal_box = internal_box
    self.fill_hole = int(fill_hole * m) if fill_hole is not None else None
    self.invert_fill_hole = invert_fill_hole
    self.exclude_colors = exclude_colors
    self.can_be_internal = can_be_internal
    self.max_distance = int(max_distance * m) if max_distance is not None else None
    self.model = None
    self.left_free = [int(x * m) for x in left_free] if left_free is not None else None
    self.right_free = [int(x * m) for x in right_free] if right_free is not None else None
    self.delete_horizontal = delete_horizontal
    self.delete_vertical = delete_vertical
    self.less_w = int(less_w * m)
    self.less_h = int(less_h * m)
    self.min_area = int(min_area * m) if min_area is not None else None
    self.max_near = int(max_near * m)
    self.n_digits = max(n_digits, 3)
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
  def create(name, file_path, newspaper_base = None, date = None, year = None, month = None, number = None, type=NEWSPAPER):
    if date is None and month is None and not type:
      file_name = Path(file_path).stem
      year = ''.join(filter(str.isdigit, file_name.split('_')[-4])) if date is not None else None
      month = MONTHS.index(file_name.split('_')[-3]) + 1 if date is not None else None
      day = file_path.split('_')[-2] if date is not None else None
      if date is not None:
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
    elif name == 'La Domenica Del Corriere':
      newspaper = La_Domenica_del_Corriere(newspaper_base, file_path, date, year, number)
    elif name == 'Corriere della Sera':
      newspaper = Corriere_della_Sera(newspaper_base, file_path, date, year, number)
    elif name == 'La Domenica':
      newspaper = La_Domenica(newspaper_base, file_path, date, year, number)
    elif name == 'Il Mondo':
      newspaper = Il_Mondo(newspaper_base, file_path, date, year, number)
    elif name == 'Il Sole 24 Ore':
      newspaper = Il_Sole_24_Ore(newspaper_base, file_path, date, year, number)
    elif name == 'Le Grandi Firme':
      newspaper = Le_Grandi_Firme(newspaper_base, file_path, date, year, number)
    elif name == 'Il Secolo Illustrato Della Domenica':
      newspaper = Il_Secolo_Illustrato(newspaper_base, file_path, date, year, number)
    elif name == 'Gazzetta Illustrata':
      newspaper = Gazzetta_Illustrata(newspaper_base, file_path, date, year, number)
    elif name == 'Italia Artistica Illustrata':
      newspaper = Italia_Artistica_Illustrata(newspaper_base, file_path, date, year, number)
    elif name == 'La Fornarina':
      newspaper = La_Fornarina(newspaper_base, file_path, date, year, number)
    elif name == 'Sfera':
      newspaper = Sfera(newspaper_base, file_path, date, year, number)
    elif name == 'La Repubblica':
      newspaper = La_Repubblica(newspaper_base, file_path, date, year, number)
    elif name == 'Il Verri':
      newspaper = Il_Verri(newspaper_base, file_path, date, year, number)
    elif name == 'Il 45':
      newspaper = Il_45(newspaper_base, file_path, date, year, number)
    elif name == 'Il Milione':
      newspaper = Il_Milione(newspaper_base, file_path, date, year, number)
    elif name == 'Campo Grafico':
      newspaper = Campo_Grafico(newspaper_base, file_path, date, year, number)
    elif name == 'Cinema Nuovo':
      newspaper = Cinema_Nuovo(newspaper_base, file_path, date, year, number)
    elif name == 'Fatto Quotidiano':
      newspaper = Fatto_Quotidiano(newspaper_base, file_path, date, year, number)
    elif name == 'Futurismo':
      newspaper = Futurismo(newspaper_base, file_path, date, year, number)
    elif name == 'Giornale Arte':
      newspaper = Giornale_Arte(newspaper_base, file_path, date, year, number)
    elif name == 'Italia Futurista':
      newspaper = Italia_Futurista(newspaper_base, file_path, date, year, number)
    elif name == 'La Lettura':
      newspaper = La_Lettura(newspaper_base, file_path, date, year, number)
    elif name == 'Lei':
      newspaper = Lei(newspaper_base, file_path, date, year, number)
    elif name == 'Officina':
      newspaper = Officina(newspaper_base, file_path, date, year, number)
    elif name == 'Pinocchio':
      newspaper = Pinocchio(newspaper_base, file_path, date, year, number)
    elif name == 'Poesia Dessy':
      newspaper = Poesia_Dessy(newspaper_base, file_path, date, year, number)
    elif name == 'Poesia Marinetti':
      newspaper = Poesia_Marinetti(newspaper_base, file_path, date, year, number)
    elif name == 'Poligono':
      newspaper = Poligono(newspaper_base, file_path, date, year, number)
    elif name == 'Politecnico':
      newspaper = Politecnico(newspaper_base, file_path, date, year, number)
    elif name == 'Prospettive':
      newspaper = Prospettive(newspaper_base, file_path, date, year, number)
    elif name == 'Pungolo della Domenica':
      newspaper = Pungolo_della_Domenica(newspaper_base, file_path, date, year, number)
    elif name == 'Questo e Altro':
      newspaper = Questo_e_Altro(newspaper_base, file_path, date, year, number)
    elif name == 'Santelia Artecrazia':
      newspaper = Santelia_Artecrazia(newspaper_base, file_path, date, year, number)
    elif name == 'Tesoretto':
      newspaper = Tesoretto(newspaper_base, file_path, date, year, number)
    elif name == 'Fiches':
      newspaper = Fiches(newspaper_base, file_path, date, year, number)
    elif name == 'Cliniche':
      newspaper = Cliniche(newspaper_base, file_path, date, year, number)
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    newspaper.month = month
    newspaper.n_page = None
    newspaper.use_ai = False
    newspaper._is_first_page = None
    newspaper.type = type
    return newspaper
  def get_page_position(self):
    return ['top']
  def get_page_ins_position(self):
    return ['top']
  def get_limits_select_images(self):
    return (10000, 15000, 5000, 100000)
  @staticmethod
  def get_start_static(name, ofset):
    if name == 'Scenario':
      parameters = Scenario.get_start(ofset)
    elif name == 'Il Mondo':
      parameters = Il_Mondo.get_start(ofset)
    elif name == 'Le Grandi Firme':
      parameters = Le_Grandi_Firme.get_start(ofset)
    elif name == 'La Domenica Del Corriere':
      parameters = La_Domenica_del_Corriere.get_start(ofset)
    elif name == 'Corriere della Sera':
      parameters = Corriere_della_Sera.get_start(ofset)
    elif name == 'La Domenica':
      parameters = La_Domenica.get_start(ofset)
    elif name == 'Il Secolo Illustrato Della Domenica':
      parameters = Il_Secolo_Illustrato.get_start(ofset)
    elif name == 'Gazzetta Illustrata':
      parameters = Gazzetta_Illustrata.get_start(ofset)
    elif name == 'Italia Artistica Illustrata':
      parameters = Italia_Artistica_Illustrata.get_start(ofset)
    elif name == 'La Fornarina':
      parameters = La_Fornarina.get_start(ofset)
    elif name == 'Sfera':
      parameters = Sfera.get_start(ofset)
    elif name == 'Il Verri':
      parameters = Il_Verri.get_start(ofset)
    elif name == 'Il 45':
      parameters = Il_45.get_start(ofset)
    elif name == 'Il Milione':
      parameters = Il_Milione.get_start(ofset)
    elif name == 'Campo Grafico':
      parameters = Campo_Grafico.get_start(ofset)
    elif name == 'Cinema Nuovo':
      parameters = Cinema_Nuovo.get_start(ofset)
    elif name == 'Fatto Quotidiano':
      parameters = Fatto_Quotidiano.get_start(ofset)
    elif name == 'Futurismo':
      parameters = Futurismo.get_start(ofset)
    elif name == 'Giornale Arte':
      parameters = Giornale_Arte.get_start(ofset)
    elif name == 'Italia Futurista':
      parameters = Italia_Futurista.get_start(ofset)
    elif name == 'La Lettura':
      parameters = La_Lettura.get_start(ofset)
    elif name == 'Lei':
      parameters = Lei.get_start(ofset)
    elif name == 'Officina':
      parameters = Officina.get_start(ofset)
    elif name == 'Pinocchio':
      parameters = Pinocchio.get_start(ofset)
    elif name == 'Poesia Dessy':
      parameters = Poesia_Dessy.get_start(ofset)
    elif name == 'Poesia Marinetti':
      parameters = Poesia_Marinetti.get_start(ofset)
    elif name == 'Poligono':
      parameters = Poligono.get_start(ofset)
    elif name == 'Politecnico':
      parameters = Politecnico.get_start(ofset)
    elif name == 'Prospettive':
      parameters = Prospettive.get_start(ofset)
    elif name == 'Pungolo della Domenica':
      parameters = Pungolo_della_Domenica.get_start(ofset)
    elif name == 'Questo e Altro':
      parameters = Questo_e_Altro.get_start(ofset)
    elif name == 'Santelia Artecrazia':
      parameters = Santelia_Artecrazia.get_start(ofset)
    elif name == 'Tesoretto':
      parameters = Tesoretto.get_start(ofset)
    elif name == 'Fiches':
      parameters = Fiches.get_start(ofset)
    elif name == 'Cliniche':
      parameters = Cliniche.get_start(ofset)
    else:
      parameters = None
    return parameters
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
    elif name == 'La Repubblica':
      parameters = La_Repubblica.get_parameters()
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    return parameters
  def get_start(self, ofset):
    return None
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
      imgs = []
      for dim in dims:
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
  def get_start_year(self):
    return 2016
  def get_head(self):
    number = self.get_number()
    year = self.init_year + self.date.year - self.get_start_year()
    if self.year_change is not None and \
        (self.year_change[1] < self.date.month or \
        (self.year_change[1] == self.date.month and self.year_change[0] <= self.date.day)):
      year += 1
    return str(year), number
  def get_page(self):
    return None, None
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    f = 1
    l = n_pages
    r = 2
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      try:
        # page.newspaper.n_page
        if page.newspaper.n_page is not None:
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
  def get_limits(self):
    return None, None, None, None
  def divide(self, img):
    imgs = []
    if img.ndim == 2:
      height, width = img.shape
    else:
      height, width, _ = img.shape
    if os.path.dirname(self.file_path).split(' ')[-1][0:2] == 'OT':
      try:
        s = os.path.dirname(self.file_path).split(' ')[-1][2:]
        if s.split('-')[0] != '':
          f = int(s.split('-')[0])
        else:
          f = int(s)
      except Exception as e:
        error = 'Folder ' + self.filedir + ' ' + 'is not valid'
        raise(error)
      if f < 0:
        f = -f
      if f == 1:
        imgs.append(img)
      elif f == 4:
        parameters = self.get_crop_parameters(0, width, height)
        img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
        parameters = self.get_crop_parameters(1, width, height)
        img2 = img[:, parameters.left:parameters.right]
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
        height, width, _ = img1.shape
        imgs.append(img1[0:height, 0:width // 2])
        imgs.append(img1[0:height, width // 2:width])
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        height, width, _ = img2.shape
        imgs.append(img2[0:height, 0:width // 2])
        imgs.append(img2[0:height, width // 2:width])
    else:
      parameters = self.get_crop_parameters(0, width, height)
      img1 = img[:, parameters.left:parameters.right]
      parameters = self.get_crop_parameters(1, width, height)
      img2 = img[:, parameters.left:parameters.right]
      imgs.append(img1)
      imgs.append(img2)
    return imgs

  # image1 = image.crop((parameters.left, parameters.top, parameters.right, parameters.bottom))
  def is_first_page(self, model = None):
    if self._is_first_page is not None:
      return self._is_first_page
    return False
  def delete_first_page(self):
    self._is_first_page = None
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
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
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
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    f = 1
    l = n_pages
    r = 2
    for n_page, page in enumerate(page_pool):
      try:
        page.newspaper.n_page
        if page.newspaper.n_page is not None:
          continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      if page.isins and page.ins_n == 2:
        page.newspaper.n_page = n_page + 1
      elif page.isins and page.ins_n == 3 and n_pages == 8:
        pg = [1,8,3,6,5,4,7,2]
        page.newspaper.n_page = pg[n_page]
      else:
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
          r = -2
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
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=50,
                                max_w=130,
                                min_h=110,
                                max_h=190,
                                ts=1)

class Alias_Domenica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 6
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Alias Domenica', file_path, date, year, number, init_page = 5)
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=50,
                                max_w=130,
                                min_h=110,
                                max_h=190,
                                ts=1)

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
    # if self.n_page is None:
    whole = [[0, 200, 1000, 650], [w - 1000, 200, w, 650]]
    # elif self.n_page % 2 == 0:
    #   whole = [[200, 200, 500, 650], [w - 450, 200, w - 150, 650]]
    # else:
    #   whole = [[200, 200, 500, 650], [w - 450, 200, w - 150, 650]]
    return whole
  def get_ins_whole_page_location(self, image):
    w, h = image.size
    # if self.n_page is None:
    whole = [[0, 200, 1000, 620], [w - 1000, 200, w, 650], [0, h - 550, 620, h],
            [w // 2 - 400, h - 600, w // 2 + 400, h - 200], [w - 1000, h - 550, w, h - 200]]
    # whole = [[100, 200, w - 100, 620], [100, h - 550, w - 100, h - 200]]
    # elif self.n_page % 2 == 0:
    #   whole = [[100, 200, 470, 620], [w - 600, 200, w - 150, 650], [100, h - 550, 620, h - 200], [w // 2 - 250, h - 600, w // 2 + 250, h - 200]]
    # else:
    #   whole = [[100, 200, 470, 620], [w - 600, 200, w - 200, 620], [w - 620, h - 550, w - 200, h - 200], [w // 2 - 250, h - 600, w // 2 + 250, h - 200]]
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    def set_n_page(page_pool, n_page, n_none):
      for page in page_pool:
        if page.newspaper.n_page is not None and page.newspaper.n_page == n_page:
          n_none += 1
          n_page = '?' + ('000' + str(n_page))[-2:] + '(' + str(n_none-1) + ')'
          break
      return n_page, n_none
    f = 1
    m = None
    n_none = 0
    if page_pool.filedir.split(' ')[-1][0] == 'P':
      try:
        f = int(page_pool.filedir.split(' ')[-1][1:])
        n_pages += f - 1
      except Exception as e:
        error = 'Folder ' + page_pool.filedir + ' ' + 'is not valid'
        raise(error)
    elif page_pool.filedir.split(' ')[-1][0:2] == 'OT':
      if use_ai:
        m = []
        min = None
        for page in page_pool:
          n_page = page.prediction
          if n_page is not None:
            if min is None:
              min = n_page
            else:
              min = n_page if n_page < min else min
        for i, page in enumerate(page_pool):
          n_page = page.prediction
          if n_page is None:
            if n_none == 0 and min is not None:
              n_none += 1
              n_page = min - 1
            else:
              n_none += 1
              n_page = '?' + ('000' + str(n_none-1))[-2:]
          m.append(n_page)
      else:
        m = [x for x in range(n_pages)]
        if n_pages == 8:
          m = [8, 1, 4, 5, 6, 3, 2, 7]
        elif n_pages == 16:
          m = [16, 1, 8, 9, 10, 7, 2, 15, 14, 3, 6, 11, 12, 5, 4, 13]
        elif n_pages == 32:
          m = [8, 25, 24, 9, 26, 7, 10, 23, 6, 27, 22, 11, 28, 5, 12, 21, 4, 29, 20, 13, 30, 3, 14, 19, 2, 31, 18, 15, 32, 1, 16, 17]
        elif n_pages == 40:
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
        page.newspaper.n_page, n_none = set_n_page(page_pool, m[n_page], n_none)
        continue
      n = Path(page.file_name).stem.split('_')[-1]
      if n != '0':
        if r == 2:
          page.newspaper.n_page, n_none = set_n_page(page_pool, f, n_none)
          f += 1
          r = -1
        elif r == -1:
          page.newspaper.n_page, n_none = set_n_page(page_pool, l, n_none)
          l -= 1
          r = -2
        elif r == -2:
          page.newspaper.n_page, n_none = set_n_page(page_pool, l, n_none)
          l -= 1
          r = 1
        elif r == 1:
          page.newspaper.n_page, n_none = set_n_page(page_pool, f, n_none)
          f += 1
          r = 2
      else:
        if page.prediction is not None and use_ai:
          if page.prediction == f:
            page.newspaper.n_page, n_none = set_n_page(page_pool, f, n_none)
            f += 1
            lasffl = 'f'
          elif page.prediction == l:
            page.newspaper.n_page, n_none = set_n_page(page_pool, l, n_none)
            l -= 1
            lasffl = 'l'
          elif str(page.prediction) in str(f):
            page.newspaper.n_page, n_none = set_n_page(page_pool, f, n_none)
            f += 1
            lasffl = 'f'
          elif str(page.prediction) in str(l):
            page.newspaper.n_page, n_none = set_n_page(page_pool, l, n_none)
            l -= 1
            lasffl = 'l'
          else:
            page.newspaper.n_page, n_none = set_n_page(page_pool, page.prediction, n_none)
        else:
          if lasffl == 'f':
            page.newspaper.n_page, n_none = set_n_page(page_pool, f, n_none)
            f += 1
          else:
            page.newspaper.n_page, n_none = set_n_page(page_pool, l, n_none)
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
                                 min_area=500,
                                 max_near=4),
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
                                 max_near=4),
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
                                 max_near=4),
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
                                 max_near=4),
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
                                 max_near=4)
            ]

  def get_page_ins_position(self):
    return ['top', 'top',' bottom', 'bottom', 'bottom']
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
                                min_area=500,
                                max_near=2,
                                n_digits=4)
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
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1932','2','--','1932','10','--')
    elif ofset == 2:
      return ('1932', '11', '--', '1933', '7', '--')
    elif ofset == 3:
      return ('1933', '8', '--', '1934', '4', '--')
    elif ofset == 4:
      return ('1934', '5', '--', '1935', '1', '--')
    elif ofset == 5:
      return ('1935', '2', '--', '1935', '12', '--')
    elif ofset == 6:
      return ('1936', '1', '--', '1932', '12', '--')
    elif ofset == 7:
      return ('1937', '1', '--', '1937', '12', '--')
    elif ofset == 8:
      return ('1938', '1', '--', '1938', '12', '--')
    elif ofset == 9:
      return ('1939', '1', '--', '1940', '5', '--')
    elif ofset == 10:
      return ('1940', '6', '--', '1941', '12', '--')
    elif ofset == 11:
      return ('1942', '1', '--', '1943', '12', '--')
    elif ofset == 12:
      return ('1949', '12', '--', '1950', '1', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 10000, 5000, 100000)
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
    self.init_year = 1
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Domenica Del Corriere', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1899','01','08','1900','09','23')
    elif ofset == 2:
      return ('1900', '09', '30', '1901', '12', '29')
    elif ofset == 3:
      return ('1902', '01', '05', '1903', '04', '29')
    elif ofset == 4:
      return ('1903', '05', '03', '1904', '09', '04')
    elif ofset == 5:
      return ('1904', '09', '11', '1906', '02', '11')
    elif ofset == 6:
      return ('1906', '02', '18', '1907', '07', '14')
    elif ofset == 7:
      return ('1907', '07', '21', '1908', '12', '27')
    elif ofset == 8:
      return ('1909', '01', '03', '1910', '06', '12')
    elif ofset == 9:
      return ('1910', '06', '19', '1911', '12', '10')
    elif ofset == 10:
      return ('1911', '12', '17', '1913', '06', '01')
    elif ofset == 11:
      return ('1913', '06', '15', '1915', '01', '03')
    elif ofset == 12:
      return ('1915', '01', '10', '1916', '05', '14')
    elif ofset == 13:
      return ('1916', '05', '21', '1917', '12', '23')
    elif ofset == 14:
      return ('1918', '01', '13', '1920', '07', '11')
    elif ofset == 15:
      return ('1920', '07', '18', '1922', '07', '30')
    elif ofset == 16:
      return ('1922', '08', '06', '1924', '01', '27')
    elif ofset == 17:
      return ('1924', '02', '03', '1925', '06', '28')
    elif ofset == 18:
      return ('1925', '07', '05', '1926', '12', '12')
    elif ofset == 19:
      return ('1926', '12', '19', '1928', '05', '20')
    elif ofset == 20:
      return ('1928', '05', '27', '1929', '11', '03')
    elif ofset == 21:
      return ('1929', '11', '10', '1931', '01', '11')
    elif ofset == 22:
      return ('1931', '01', '18', '1932', '03', '13')
    elif ofset == 23:
      return ('1932', '03', '13', '1933', '04', '23')
    elif ofset == 24:
      return ('1933', '04', '30', '1934', '06', '17')
    elif ofset == 25:
      return ('1934', '06', '24', '1935', '08', '11')
    elif ofset == 26:
      return ('1935', '08', '18', '1937', '01', '10')
    elif ofset == 27:
      return ('1937', '01', '17', '1938', '03', '20')
    elif ofset == 28:
      return ('1938', '03', '27', '1939', '05', '13')
    elif ofset == 29:
      return ('1939', '05', '20', '1940', '09', '01')
    elif ofset == 30:
      return ('1940', '09', '18', '1942', '02', '08')
    elif ofset == 31:
      return ('1942', '02', '15', '1943', '12', '26')
    elif ofset == 32:
      return ('1944', '01', '02', '1946', '08', '18')
    elif ofset == 33:
      return ('1944', '01', '02', '1948', '01', '25')
    elif ofset == 34:
      return ('1948', '02', '01', '1949', '04', '10')
    elif ofset == 35:
      return ('1949', '04', '17', '1950', '05', '14')
    elif ofset == 36:
      return ('1950', '05', '21', '1950', '12', '31')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = ((w + 100) // 2, 200 + 800, w - 800, 1700)
    return whole
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    for n_page, page in enumerate(page_pool):
      # try:
      #   page.newspaper.n_page
      #   continue
      # except:
      #   pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page + 1
  def get_start_year(self):
    return 1899
  def get_remove_borders_parameters(self, i, width, height):
    left = 950
    top = 800
    right = width - 800
    bottom = height - 900
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def get_crop_parameters(self, i, width, height):
    if i == 0:
      left = width // 2 - 50
      top = 0
      right = width
      bottom = height
    elif i == 1:
      left = 0
      top = 0
      right = width // 2 + 50
      bottom = height
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def is_first_page(self, img=None, model=None, get_crop=False):
    if self._is_first_page  is not None and not get_crop:
      return self._is_first_page
    if img is None or model is None and not get_crop:
      return None
    oh, ow, _ = img.shape
    if ow > oh:
      crop = img[0:1200, ow // 2:, :]
    else:
      crop = img[0:1200, :, :]
    crop = cv2.resize(crop, (224, 224), Image.Resampling.LANCZOS)
    cv2.imwrite(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    crop = Image.open(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'))
    if self._is_first_page  is not None and get_crop:
      return self._is_first_page, np.array(crop)
    dataset = []
    crop = tf.image.convert_image_dtype(crop, dtype=tf.float32)
    dataset.append(crop)
    try:
      predictions = list(np.argmax(model.predict(np.array(dataset), verbose=0), axis=-1))
    except Exception as e:
      predictions[0] = 1
    self._is_first_page = predictions[0] == 0
    if get_crop:
      return self._is_first_page, np.array(crop)
    return self._is_first_page
  def get_ofset(self):
    return 0, -200, 0, 200
  def get_dimension(self, img=None):
    return 5600, 7400
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def divide(self, img):
    imgs = []
    height, width, _ = img.shape
    parameters = self.get_crop_parameters(1, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    parameters = self.get_crop_parameters(0, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    imgs.append(img1)
    imgs.append(img2)
    return imgs
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
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (11000, 8000, 500, 500)
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
  def is_first_page(self, img=None, model=None, get_crop=False):
    if self._is_first_page  is not None and not get_crop:
      return self._is_first_page
    if img is None or model is None and not get_crop:
      return None
    oh, ow, _ = img.shape
    if ow > oh:
      crop = img[0:800, ow // 2:, :]
    else:
      crop = img[0:800, :, :]
    crop = cv2.resize(crop, (224, 224), Image.Resampling.LANCZOS)
    cv2.imwrite(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    crop = Image.open(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'))
    if self._is_first_page  is not None and get_crop:
      return self._is_first_page, np.array(crop)
    dataset = []
    crop = tf.image.convert_image_dtype(crop, dtype=tf.float32)
    dataset.append(crop)
    try:
      predictions = list(np.argmax(model.predict(np.array(dataset), verbose=0), axis=-1))
    except Exception as e:
      predictions[0] = 1
    self._is_first_page = predictions[0] == 0
    if get_crop:
      return self._is_first_page, np.array(crop)
    return self._is_first_page
  def divide(self, img):
    imgs = []
    height, width = img.shape
    parameters = self.get_crop_parameters(1, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right + 20]
    parameters = self.get_crop_parameters(0, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left - 20:parameters.right]
    imgs.append(img1)
    imgs.append(img2)
    return imgs
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1949', '02', '19', '1949', '12', '31')
    elif ofset == 2:
      return ('1950', '01', '--', '1950', '12', '31')
    elif ofset == 3:
      return ('1951', '01', '--', '1952', '12', '31')
    elif ofset == 4:
      return ('1953', '01', '03', '1953', '12', '29')
    elif ofset == 5:
      return ('1954', '01', '05', '1954', '12', '28')
    elif ofset == 6:
      return ('1955', '01', '--', '1956', '12', '--')
    elif ofset == 7:
      return ('1957', '01', '--', '1957', '12', '--')
    elif ofset == 8:
      return ('1958', '01', '--', '1959', '12', '--')
    elif ofset == 9:
      return ('1960', '01', '--', '1961', '11', '10')
    elif ofset == 10:
      return ('1961', '11', '14', '1961', '12', '--')
    elif ofset == 11:
      return ('1962', '01', '02', '1962', '12', '25')
    elif ofset == 12:
      return ('1963', '01', '--', '1963', '06', '25')
    elif ofset == 13:
      return ('1963', '06', '11', '1963', '12', '--')
    elif ofset == 14:
      return ('1964', '01', '--', '1964', '12', '--')
    elif ofset == 15:
      return ('1965', '01', '--', '1965', '12', '--')
    elif ofset == 16:
      return ('1966', '01', '04', '1966', '03', '08')
    elif ofset == 17:
      return ('1969', '10', '30', '1969', '11', '25')
    elif ofset == 18:
      return ('1970', '01', '--', '1970', '06', '--')
    elif ofset == 19:
      return ('1970', '07', '05', '1970', '12', '27')
    elif ofset == 20:
      return ('1971', '01', '03', '1971', '11', '28')
    elif ofset == 21:
      return ('1971', '12', '05', '1971', '12', '26')
    elif ofset == 22:
      return ('1972', '09', '07', '1972', '12', '07')
    elif ofset == 23:
      return ('1972', '05', '05', '1972', '08', '31')
    elif ofset == 24:
      return ('1973', '05', '03', '1973', '12', '27')
    elif ofset == 25:
      return ('1973', '01', '11', '1973', '04', '26')
    elif ofset == 26:
      return ('1972', '01', '07', '1972', '04', '28')
    elif ofset == 27:
      return ('1974', '01', '10', '1974', '08', '29')
    elif ofset == 28:
      return ('1974', '09', '05', '1975', '12', '26')
    elif ofset == 29:
      return ('1975', '01', '09', '1975', '02', '27')
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

class Le_Grandi_Firme(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Mondo', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  def get_limits(self):
    return (11000, 8000, 500, 500)
  def divide(self, img):
    imgs = []
    height, width = img.shape
    parameters = self.get_crop_parameters(1, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right + 20]
    parameters = self.get_crop_parameters(0, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left - 20:parameters.right]
    imgs.append(img1)
    imgs.append(img2)
    return imgs
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1924', '09', '16', '1924', '12', '--')
    elif ofset == 2:
      return ('1925', '01', '--', '1925', '12', '--')
    elif ofset == 3:
      return ('1926', '01', '--', '1926', '12', '--')
    elif ofset == 4:
      return ('1927', '01', '--', '1927', '12', '--')
    elif ofset == 5:
      return ('1928', '02', '--', '1928', '03', '--')
    elif ofset == 6:
      return ('1929', '01', '--', '1929', '12', '--')
    elif ofset == 7:
      return ('1930', '01', '--', '1930', '12', '--')
    elif ofset == 8:
      return ('1931', '01', '--', '1931', '12', '--')
    elif ofset == 9:
      return ('1932', '01', '--', '1932', '12', '--')
    elif ofset == 10:
      return ('1933', '11', '--', '1935', '12', '--')
    elif ofset == 11:
      return ('1937', '01', '--', '1937', '12', '--')
    elif ofset == 12:
      return ('1938', '01', '--', '1938', '10', '--')
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


class Il_Secolo_Illustrato(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Secolo Illustrato Della Domenica', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (11000, 8000, 500, 500)
  def divide(self, img):
    imgs = []
    height, width = img.shape
    parameters = self.get_crop_parameters(1, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right + 20]
    parameters = self.get_crop_parameters(0, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left - 20:parameters.right]
    imgs.append(img1)
    imgs.append(img2)
    return imgs
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1889', '10', '--', '1891', '12', '--')
    elif ofset == 2:
      return ('1892', '01', '--', '1892', '12', '--')
    elif ofset == 3:
      return ('1893', '01', '--', '1893', '12', '--')
    elif ofset == 4:
      return ('1894', '01', '--', '1896', '12', '--')
    elif ofset == 5:
      return ('1897', '01', '--', '1897', '12', '--')
    elif ofset == 6:
      return ('1898', '01', '--', '1899', '12', '--')
    elif ofset == 7:
      return ('1900', '01', '--', '1901', '12', '--')
    elif ofset == 8:
      return ('1902', '01', '--', '1903', '12', '--')
    elif ofset == 9:
      return ('1904', '01', '--', '1905', '12', '--')
    elif ofset == 10:
      return ('1906', '11', '--', '1907', '12', '--')
    elif ofset == 11:
      return ('1908', '01', '--', '1937', '12', '--')
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


class La_Domenica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 1
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Domenica', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1899','01','08','1900','09','23')
    elif ofset == 2:
      return ('1900', '09', '30', '1901', '12', '29')
    elif ofset == 3:
      return ('1902', '01', '05', '1903', '04', '29')
    elif ofset == 4:
      return ('1903', '05', '03', '1904', '09', '04')
    elif ofset == 5:
      return ('1904', '09', '11', '1906', '02', '11')
    elif ofset == 6:
      return ('1906', '02', '18', '1907', '07', '14')
    elif ofset == 7:
      return ('1907', '07', '21', '1908', '12', '27')
    elif ofset == 8:
      return ('1909', '01', '03', '1910', '06', '12')
    elif ofset == 9:
      return ('1910', '06', '19', '1911', '12', '10')
    elif ofset == 10:
      return ('1911', '12', '17', '1913', '06', '01')
    elif ofset == 11:
      return ('1913', '06', '15', '1915', '01', '03')
    elif ofset == 12:
      return ('1915', '01', '10', '1916', '05', '14')
    elif ofset == 13:
      return ('1916', '05', '21', '1917', '12', '23')
    elif ofset == 14:
      return ('1918', '01', '13', '1920', '07', '11')
    elif ofset == 15:
      return ('1920', '07', '18', '1922', '07', '30')
    elif ofset == 16:
      return ('1922', '08', '06', '1924', '01', '27')
    elif ofset == 17:
      return ('1924', '02', '03', '1925', '06', '28')
    elif ofset == 18:
      return ('1925', '07', '05', '1926', '12', '12')
    elif ofset == 19:
      return ('1926', '12', '19', '1928', '05', '20')
    elif ofset == 20:
      return ('1928', '05', '27', '1929', '11', '03')
    elif ofset == 21:
      return ('1929', '11', '10', '1931', '01', '11')
    elif ofset == 22:
      return ('1931', '01', '18', '1932', '03', '13')
    elif ofset == 23:
      return ('1932', '03', '13', '1933', '04', '23')
    elif ofset == 24:
      return ('1933', '04', '30', '1934', '06', '17')
    elif ofset == 25:
      return ('1934', '06', '24', '1935', '08', '11')
    elif ofset == 26:
      return ('1935', '08', '18', '1937', '01', '10')
    elif ofset == 27:
      return ('1937', '01', '17', '1938', '03', '20')
    elif ofset == 28:
      return ('1938', '03', '27', '1939', '05', '13')
    elif ofset == 29:
      return ('1939', '05', '20', '1940', '09', '01')
    elif ofset == 30:
      return ('1940', '09', '18', '1942', '02', '08')
    elif ofset == 31:
      return ('1942', '02', '15', '1943', '12', '26')
    elif ofset == 32:
      return ('1944', '01', '02', '1946', '08', '18')
    elif ofset == 33:
      return ('1944', '01', '02', '1948', '01', '25')
    elif ofset == 34:
      return ('1948', '02', '01', '1949', '04', '10')
    elif ofset == 35:
      return ('1949', '04', '17', '1950', '05', '14')
    elif ofset == 36:
      return ('1950', '05', '21', '1950', '12', '31')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = ((w + 100) // 2, 200 + 800, w - 800, 1700)
    return whole
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    for n_page, page in enumerate(page_pool):
      # try:
      #   page.newspaper.n_page
      #   continue
      # except:
      #   pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page + 1
  def get_start_year(self):
    return 1899
  def get_remove_borders_parameters(self, i, width, height):
    left = 950
    top = 800
    right = width - 800
    bottom = height - 900
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def get_crop_parameters(self, i, width, height):
    if i == 0:
      left = width // 2 - 50
      top = 0
      right = width
      bottom = height
    elif i == 1:
      left = 0
      top = 0
      right = width // 2 + 50
      bottom = height
    return Newspaper_crop_parameters(left,
                                     right,
                                     top,
                                     bottom)
  def is_first_page(self, img=None, model=None, get_crop=False):
    if self._is_first_page  is not None and not get_crop:
      return self._is_first_page
    if img is None or model is None and not get_crop:
      return None
    oh, ow, _ = img.shape
    if ow > oh:
      crop = img[0:1200, ow // 2:, :]
    else:
      crop = img[0:1200, :, :]
    crop = cv2.resize(crop, (224, 224), Image.Resampling.LANCZOS)
    cv2.imwrite(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    crop = Image.open(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'))
    if self._is_first_page  is not None and get_crop:
      return self._is_first_page, np.array(crop)
    dataset = []
    crop = tf.image.convert_image_dtype(crop, dtype=tf.float32)
    dataset.append(crop)
    try:
      predictions = list(np.argmax(model.predict(np.array(dataset), verbose=0), axis=-1))
    except Exception as e:
      predictions[0] = 1
    self._is_first_page = predictions[0] == 0
    if get_crop:
      return self._is_first_page, np.array(crop)
    return self._is_first_page
  def get_ofset(self):
    return 0, -200, 0, 200
  def get_dimension(self, img=None):
    return 5600, 7400
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def divide(self, img):
    imgs = []
    height, width, _ = img.shape
    parameters = self.get_crop_parameters(1, width, height)
    img1 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    parameters = self.get_crop_parameters(0, width, height)
    img2 = img[parameters.top:parameters.bottom, parameters.left:parameters.right]
    imgs.append(img1)
    imgs.append(img2)
    return imgs
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

class Gazzetta_Illustrata(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Gazzetta Illustrata', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1877','2','18','1877','12','30')
    elif ofset == 2:
      return ('1878', '1', '--', '1878', '12', '--')
    elif ofset == 3:
      return ('1879', '1', '--', '1879', '12', '--')
    elif ofset == 4:
      return ('1882', '12', '24', '1883', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, h)
    return whole
  @staticmethod
  def get_parameters():
    return None
class Italia_Artistica_Illustrata(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Italia Artistica Illustrata', file_path, date, year, number, init_page=3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1884', '1', '27', '1884', '12', '30')
    elif ofset == 2:
      return ('1885', '1', '--', '1885', '12', '--')
    elif ofset == 3:
      return ('1886', '1', '--', '1886', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, h)
    return whole
  @staticmethod
  def get_parameters():
    return None

class La_Fornarina(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Fornarina', file_path, date, year, number, init_page=3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1882', '5', '28', '1882', '12', '--')
    elif ofset == 2:
      return ('1883', '1', '--', '1883', '5', '20')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, h)
    return whole
  @staticmethod
  def get_parameters():
    return None

class Sfera(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Sfera', file_path, date, year, number, init_page=3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1946', '11', '--', '1946', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, h)
    return whole
  @staticmethod
  def get_parameters():
    return None

class La_Repubblica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 93
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Repubblica', file_path, date, year, number, init_page=5)
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, h - 700, w, h - 300)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    f = 1
    l = n_pages
    r = 2
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      try:
        # page.newspaper.n_page
        if page.newspaper.n_page is not None:
          continue
      except:
        pass
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      n = Path(page.file_name).stem.split('_')[-1]
      if page.isins:
        page.newspaper.n_page = n_page + 1
      else:
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

class Corriere_della_Sera(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 17
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Corriere della Sera', file_path, date, year, number, init_page=3)

  def get_whole_page_location(self, image):
    w, h = image.size
    if self.n_page % 2 == 0:
      whole = [0, 150, 1000, 500]
    else:
      whole = [w - 1000, 150, w, 500]
    return whole

  @staticmethod
  def get_parameters():
    return Newspaper_parameters(scale=200,
                                min_w=10,
                                max_w=120,
                                min_h=90,
                                max_h=150,
                                ts=240,
                                min_mean=0,
                                max_mean=500,
                                invert=True,
                                max_distance=10,
                                can_be_internal=True)

class Il_Verri(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 59
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Verri', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1956','10','--','1959','10','--')
    elif ofset == 2:
      return ('1959', '12', '--', '1961', '12', '--')
    elif ofset == 3:
      return ('1962', '1', '--', '1963', '12', '--')
    elif ofset == 4:
      return ('1964', '1', '--', '1966', '12', '--')
    elif ofset == 5:
      return ('1967', '1', '--', '1971', '12', '--')
    elif ofset == 6:
      return ('1972', '1', '--', '1974', '12', '--')
    elif ofset == 7:
      return ('1975', '1', '--', '1977', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (4000, 6000, 2000, 6000)
  @staticmethod
  def get_parameters():
    return None


class Il_45(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il 45', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1945','1','--','1946','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Il_Milione(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Milione', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1938','10','13','1938','10','13')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Campo_Grafico(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Campo Grafico', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1933','01','--','1936','11','--')
    elif ofset == 2:
      return ('1937', '01', '--', '1939', '09', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Cinema_Nuovo(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Cinema Nuovo', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1952','12','--','1953','06','--')
    elif ofset == 2:
      return ('1953', '07', '--', '1953', '12', '--')
    elif ofset == 3:
      return ('1954', '01', '--', '1954', '12', '--')
    elif ofset == 4:
      return ('1955', '01', '--', '1955', '06', '--')
    elif ofset == 5:
      return ('1955', '07', '--', '1955', '12', '--')
    elif ofset == 6:
      return ('1956', '01', '--', '1956', '12', '--')
    elif ofset == 7:
      return ('1957', '01', '--', '1957', '12', '--')
    elif ofset == 8:
      return ('1958', '01', '--', '1958', '06', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Fatto_Quotidiano(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Fatto Quotidiano', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('2009','09','--','2009','12','--')
    elif ofset == 2:
      return ('2010', '01', '--', '2010', '06', '--')
    elif ofset == 3:
      return ('2010', '07', '--', '2010', '12', '--')
    elif ofset == 4:
      return ('2011', '01', '--', '2011', '06', '--')
    elif ofset == 5:
      return ('2011', '06', '--', '2011', '12', '--')
    elif ofset == 6:
      return ('2012', '01', '--', '2012', '06', '--')
    elif ofset == 7:
      return ('2012', '06', '--', '2012', '12', '--')
    elif ofset == 8:
      return ('2013', '01', '--', '2013', '06', '--')
    elif ofset == 9:
      return ('2013', '06', '--', '2013', '12', '--')
    elif ofset == 10:
      return ('2014', '01', '--', '2014', '06', '--')
    elif ofset == 11:
      return ('2014', '06', '--', '2014', '12', '--')
    elif ofset == 12:
      return ('2015', '01', '01', '2015', '01', '31')
    elif ofset == 13:
      return ('2015', '02', '01', '2015', '02', '31')
    elif ofset == 14:
      return ('2015', '03', '01', '2015', '03', '31')
    elif ofset == 15:
      return ('2015', '04', '01', '2015', '05', '31')
    elif ofset == 16:
      return ('2015', '06', '01', '2015', '06', '31')
    elif ofset == 17:
      return ('2015', '07', '01', '2015', '07', '31')
    elif ofset == 18:
      return ('2015', '08', '01', '2015', '08', '31')
    elif ofset == 19:
      return ('2015', '09', '01', '2015', '09', '31')
    elif ofset == 20:
      return ('2015', '10', '01', '2015', '10', '31')
    elif ofset == 21:
      return ('2015', '11', '01', '2015', '11', '31')
    elif ofset == 22:
      return ('2015', '12', '01', '2015', '12', '31')

  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (3000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Futurismo(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Futurismo', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1932','01','--','1933','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Giornale_Arte(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Giornale Arte', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1929','01','--','1948','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Italia_Futurista(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Italia Futurista', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1916','01','01','1918','02','11')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class La_Lettura(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Lettura', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1901','01','--','1901','06','--')
    elif ofset == 2:
      return ('1901', '07', '--', '1903', '11', '--')
    elif ofset == 3:
      return ('1903', '12', '--', '1904', '12', '--')
    elif ofset == 4:
      return ('1905', '01', '--', '1905', '12', '--')
    elif ofset == 5:
      return ('1906', '01', '--', '1907', '04', '--')
    elif ofset == 6:
      return ('1907', '05', '--', '1908', '09', '--')
    elif ofset == 7:
      return ('1908', '10', '--', '1910', '04', '--')
    elif ofset == 8:
      return ('1910', '05', '--', '1911', '10', '--')
    elif ofset == 9:
      return ('1911', '11', '--', '1913', '06', '--')
    elif ofset == 10:
      return ('1913', '07', '--', '1914', '12', '--')
    elif ofset == 11:
      return ('1915', '01', '--', '1916', '06', '--')
    elif ofset == 12:
      return ('1916', '07', '--', '1917', '12', '--')
    elif ofset == 13:
      return ('1918', '01', '--', '1919', '06', '--')
    elif ofset == 14:
      return ('1919', '07', '--', '1920', '12', '--')
    elif ofset == 15:
      return ('1921', '01', '--', '1922', '09', '--')
    elif ofset == 16:
      return ('1922', '10', '--', '1924', '06', '--')
    elif ofset == 17:
      return ('1924', '07', '--', '1926', '03', '--')
    elif ofset == 18:
      return ('1926', '04', '--', '1927', '12', '--')
    elif ofset == 19:
      return ('1928', '01', '--', '1929', '08', '--')
    elif ofset == 20:
      return ('1929', '09', '--', '1930', '12', '--')
    elif ofset == 21:
      return ('1931', '01', '--', '1932', '06', '--')
    elif ofset == 22:
      return ('1932', '07', '--', '1933', '12', '--')
    elif ofset == 23:
      return ('1934', '01', '--', '1935', '06', '--')
    elif ofset == 24:
      return ('1935', '07', '--', '1936', '12', '--')
    elif ofset == 25:
      return ('1937', '01', '--', '1938', '05', '--')
    elif ofset == 26:
      return ('1938', '06', '--', '1939', '12', '--')
    elif ofset == 27:
      return ('1940', '01', '--', '1941', '05', '--')
    elif ofset == 28:
      return ('1941', '06', '--', '1942', '06', '--')
    elif ofset == 29:
      return ('1942', '07', '--', '1943', '12', '--')
    elif ofset == 30:
      return ('1944', '01', '--', '1952', '10', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Lei(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Lei', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1933','07','15','1934','12','25')
    elif ofset == 2:
      return ('1935', '01', '01', '1936', '12', '29')
    elif ofset == 3:
      return ('1937', '01', '05', '1938', '12', '27')
    elif ofset == 4:
      return ('1939', '01', '03', '1940', '12', '31')
    elif ofset == 5:
      return ('1941', '01', '07', '1942', '12', '29')
    elif ofset == 6:
      return ('1943', '01', '05', '1944', '07', '18')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (12000, 100000, 8000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Officina(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Officina', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1955','01','--','1959','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Pinocchio(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Pinocchio', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1938','10','13','1938','10','13')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Poesia_Dessy(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Poesia Dessy', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1920','01','--','1920','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (12000, 100000, 4000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Poesia_Marinetti(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Poesia Marinetti', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1905','01','--','1905','11','--')
    elif ofset == 2:
      return ('1906','01','--','1909','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Poligono(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Poligono', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1927','10','--','1929','03','--')
    elif ofset == 2:
      return ('1929', '11', '--', '1931', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Politecnico(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Politecnico', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1945','01','--','1947','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Prospettive(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Prospettive', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1936','01','--','1940','12','--')
    elif ofset == 2:
      return ('1941', '01', '--', '1941', '12', '--')
    elif ofset == 3:
      return ('1942', '12', '--', '1943', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Pungolo_della_Domenica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Pungolo della Domenica', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1883','02','04','1984','12','21')
    elif ofset == 2:
      return ('1885', '01', '04', '1885', '12', '27')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Questo_e_Altro(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Questo e Altro', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1962','01','--','1964','08','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Santelia_Artecrazia(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Santelia Artecrazia', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1934','01','--','1934','12','--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Tesoretto(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Tesoretto', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1939','01','--','1940','12','--')
    elif ofset == 2:
      return ('1941', '01', '--', '1942', '12', '--')
    elif ofset == 3:
      return ('1945', '01', '--', '1945', '12', '--')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Fiches(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Fiches', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('741', '2339')
    elif ofset == 2:
      return ('741', '2341')
    elif ofset == 3:
      return ('741', '2342')
    elif ofset == 4:
      return ('741', '2362')
    elif ofset == 5:
      return ('741', '2363')
    elif ofset == 6:
      return ('741', '2364')
    elif ofset == 7:
      return ('741', '2365')
    elif ofset == 8:
      return ('741', '2411')
    elif ofset == 9:
      return ('741', '2412')
    elif ofset == 10:
      return ('741', '2413')
    elif ofset == 11:
      return ('741', '2414')
    elif ofset == 12:
      return ('741', '2415')
    elif ofset == 13:
      return ('741', '2431')
    elif ofset == 14:
      return ('741', '2442')
    elif ofset == 15:
      return ('741', '2443')
    elif ofset == 16:
      return ('741', '2444')
    elif ofset == 17:
      return ('741', '2561')
    elif ofset == 18:
      return ('741', '2563')
    elif ofset == 19:
      return ('741', '2564')
    elif ofset == 20:
      return ('741', '2565')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (5000, 100000, 3000, 100000)
  @staticmethod
  def get_parameters():
    return None

class Cliniche(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 45
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Cliniche', file_path, date, year, number, init_page = 3)
    self.contrast = 50
  def get_number(self):
    return None
  def get_head(self):
    return None, None
  @staticmethod
  def get_start(ofset):
    if ofset == 1:
      return ('1391', '1529')
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 0, w, 700)
    return whole
  def set_n_pages(self, page_pool, n_pages, use_ai=False):
    l = n_pages
    count_zero = 0
    for n_page, page in enumerate(page_pool):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(page_pool)
      page.newspaper.n_page = n_page
  def get_limits(self):
    return (8800, 6000, 1500, 1000)
  def get_limits_select_images(self):
    return (1000, 100000, 1000, 100000)
  @staticmethod
  def get_parameters():
    return None