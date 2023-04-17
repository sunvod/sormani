from __future__ import annotations

import pathlib
import cv2
import os
import numpy as np
import datetime
import time

from PyPDF2 import PdfFileMerger
from scipy.spatial import KDTree, distance

from multiprocessing import Pool, Manager

from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageTk
import tkinter as tk
from tkinter import Label, Button, RAISED

import warnings
warnings.filterwarnings("ignore")

class Page:
  def __init__(self, file_name, date, newspaper, isins, original_image, pdf_path, jpg_path, txt_path, model, debug=False):
    self.original_file_name = file_name
    self.file_name = file_name
    self.original_image = original_image
    self.original_path = str(Path(original_image).parent.resolve())
    self.date = date
    self.year = date.year
    self.month = date.month
    self.month_text = MONTHS[self.month - 1]
    self.day = date.day
    self.isins = isins
    self.newspaper = Newspaper.create(newspaper.name, original_image, newspaper.newspaper_base, date, newspaper.year, newspaper.number, model=model)
    self.pdf_path = pdf_path
    self.pdf_file_name = os.path.join(self.pdf_path, 'pdf', self.file_name) + '.pdf'
    self.jpg_path = jpg_path
    self.txt_path = txt_path
    self.txt_file_name = os.path.join(txt_path, self.file_name) + '.txt'
    self.original_txt_file_name = self.txt_file_name
    self.conversions = []
    self.page_control = -1
    self.is_bobina = False
    self.isdivided = False
    self.model = model
    self.prediction = None
    self.debug = debug
  def add_conversion(self, conversion):
    if isinstance(conversion, list):
      for conv in conversion:
        self.conversions.append(conv)
    else:
      self.conversions.append(conversion)
  def set_file_names(self):
    if str(self.newspaper.n_page).isdigit():
      parameters = self.newspaper.get_parameters()
      page = ('00000' + str(self.newspaper.n_page))[-parameters.n_digits : ]
    else:
      page = self.newspaper.n_page
    #page = ('000' + str(self.newspaper.n_page))[-3:]
    self.file_name = self.newspaper.name.replace(' ', '_') \
                     + '_' + str(self.year) \
                     + '_' + str(self.month_text) \
                     + '_' + (str(self.day) if self.day >= 10 else '0' + str(self.day)) \
                     + '_p' + page
    self.txt_file_name = os.path.join(self.txt_path, self.file_name) + '.txt'
  def change_contrast(self):
    def isgray(image):
      img = np.asarray(image)
      if len(img.shape) < 3:
        return True
      if img.shape[2] == 1:
        return True
      r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
      if np.allclose(r, g) and np.allclose(r, b):
        return True
      return False
    if self.force or not self.isAlreadySeen():
      try:
        contrast = self.contrast if self.contrast is not None else self.newspaper.contrast
        if contrast is None:
          return 0
        image = Image.open(self.original_image)
        # pixel_map = image.load()
        pixel_map = (np.array(image.crop([0,0,1,1])), np.array(image.crop([image.size[0] - 1, image.size[1] - 1, image.size[0], image.size[1]])))
        if not isgray(pixel_map):
          if (pixel_map[0] == (64, 62, 22)).all() and (pixel_map[1] == (64, 62, 22)).all():
            return 0
        else:
          if (pixel_map[0] == (64, 0, 0)).all() and (pixel_map[1] == (62, 0, 0)).all():
            return 0
        image = self._change_contrast(image, contrast)
        pixel_map = image.load()
        if not isgray(pixel_map):
          pixel_map[0, 0] = (64, 62, 22)
          pixel_map[image.size[0] - 1, image.size[1] - 1] = (64, 62, 22)
        else:
          pixel_map[0, 0] = (64)
          pixel_map[image.size[0] - 1, image.size[1] - 1] = (62)
        image.save(self.original_image)
        return 1
      except Exception as e:
        pass
    return 0
  def _change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def change_threshold(self):
      img = cv2.imread(self.original_image)
      if self.inversion:
        ret, img = cv2.threshold(img, self.limit, self.color, cv2.THRESH_BINARY_INV)
      else:
        ret, img = cv2.threshold(img, self.limit, self.color, cv2.THRESH_BINARY)
      cv2.imwrite(self.original_image, img)
      return 1
  def change_colors(self):
      img = cv2.imread(self.original_image)
      if self.inversion:
        img[img <= int(self.limit, 16)] = 255 - self.color
      else:
        img[img >= int(self.limit, 16)] = self.color
      cv2.imwrite(self.original_image, img)
      return 1
  def improve_images(self):
      count = 0
      file = self.original_image
      file_img = '.'.join(file.split('.')[:-1]) + '_img.' + file.split('.')[-1]
      file_bing = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
      file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
      # Questo riempie i buchi neri
      fill_hole = 4
      invert_fill_hole = True
      if invert_fill_hole:
        thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      else:
        thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((fill_hole, fill_hole), np.uint8)
      thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=16)
      if invert_fill_hole:
        thresh = 255 - thresh
      # Questo riempie i buchi bianchi
      fill_hole = 8
      invert_fill_hole = False
      if invert_fill_hole:
        thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      else:
        thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((fill_hole, fill_hole), np.uint8)
      thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=16)
      if invert_fill_hole:
        thresh = 255 - thresh
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      nimg = img.copy()
      if self.debug:
        bimg = img.copy()
        bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
      white_nimg = 255 - np.zeros_like(nimg)
      for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > self.limit and h > self.limit and hierarchy[0, i, 3] >= 3:
          cv2.rectangle(nimg,(x, y), (x + w, y + h), 0, -1)
          cv2.rectangle(white_nimg, (x, y), (x + w, y + h), 0, -1)
          if self.debug:
            cv2.drawContours(bimg, contour, -1, (0, 255, 0), 3)
          count += 1
      nimg[nimg >= int(self.threshold, 16)] = 255
      nimg[nimg < int(self.threshold, 16)] = 0
      nimg[white_nimg == 0] = img[white_nimg == 0]
      if not self.debug:
        cv2.imwrite(file, nimg)
      else:
        cv2.imwrite(file_img, nimg)
        cv2.imwrite(file_bing, bimg)
        cv2.imwrite(file_thresh, white_nimg)
      return count
  def save_pages_images(self, storage):
    if self.isAlreadySeen():
      pos = self.newspaper.get_whole_page_location()
      image = Image.open(self.original_image)
      image = image.crop(pos)
      image = image.resize(((int)(image.size[0] * 1.5), (int)(image.size[1] * 1.5)), Image.Resampling.LANCZOS)
      n_files = sum(1 for _, _, files in os.walk(storage) for f in files)
      file_count = str('000000000' + str(n_files))[-7:]
      file_name = os.path.join(storage, file_count + '_' + self.file_name) + pathlib.Path(self.original_image).suffix
      image.save(file_name)
      return True
    return False
  def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
    return Image.fromarray(img)
  def convert_from_image_to_cv2(self, image: Image) -> np.ndarray:
    return np.asarray(image)
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

  def get_near_contour(self, img, file_name, contours, pos, parameters):
    i = pos
    found_left = []
    contour = contours[pos]
    while i > 0:
      _contour = contours[i - 1]
      x, y, w, h = cv2.boundingRect(contour)
      _x, _y, _w, _h = cv2.boundingRect(_contour)
      if self.debug:
        _roi = img[_y:_y + _h, _x:_x + _w]
        name = file_name + '_' + '00000' + str(i)[-5:]
        filedir = os.path.join(STORAGE_BASE, REPOSITORY)
        filedir += '_' + self.newspaper.name.lower().replace(' ', '_')
        Path(filedir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(filedir, name + '_left.jpg'), _roi)
      if _x < x and _y + _h > y + h - h // 2 and _y + _h < y + h + h // 2 and _w > parameters.less_w and _h > parameters.less_h and _w * _h > w * h // 3:
        found_left.append(_contour)
        contour = _contour
      if len(found_left) >= 16:
        break
      i -= 1
    i = pos
    if i:
      i -= 1
    found_right = []
    contour = contours[pos]
    while i < len(contours):
      if i == pos:
        i += 1
        continue
      _contour = contours[i]
      x, y, w, h = cv2.boundingRect(contour)
      _x, _y, _w, _h = cv2.boundingRect(_contour)
      if _x == x and _w > w:
        points = [(x + w + 1, y), (x + _w, y), (x + _w, y + _h), (x + w, y + _h)]
        _contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
        _x = x + w + 1
      if self.debug:
        if _x > x:
          _roi = img[_y:_y + _h, _x:_x + _w]
          name = file_name + '_' + '00000' + str(i)[-5:]
          filedir = os.path.join(STORAGE_BASE, REPOSITORY)
          filedir += '_' + self.newspaper.name.lower().replace(' ', '_')
          Path(filedir).mkdir(parents=True, exist_ok=True)
          cv2.imwrite(os.path.join(filedir, name + '_right.jpg'), _roi)
      if _x > x and _y + _h > y + h - h // 2 and _y + _h < y + h + h // 2 and _w > parameters.less_w and _h > parameters.less_h:
        found_right.append(_contour)
        contour = _contour
      if len(found_right) >= 16:
        break
      i += 1
    return found_left, found_right
  def clean_contours(self, img, file_name, contours, hierarchy, parameters):
    def _ordering_contours_x(c):
      return cv2.boundingRect(c)[0]
    contours.sort(key=_ordering_contours_x)
    _contours = []
    heigth, width = img.shape
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      if (not parameters.can_be_internal and hierarchy[0, i, 3] != -1) \
          or w <= parameters.box[0] \
          or w >= parameters.box[1] \
          or h <= parameters.box[2] \
          or h >= parameters.box[3] \
          or w > h * 1.1 \
          or x == 0 \
          or y == 0 \
          or x + w == width \
          or y + h == heigth:
        continue
      if self.debug:
        _roi = img[y:y + h, x:x + w]
        name = file_name + '_' + '00000' + str(i)[-5:]
        filedir = os.path.join(STORAGE_BASE, REPOSITORY)
        filedir += '_' + self.newspaper.name.lower().replace(' ', '_')
        Path(filedir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(filedir, name + '_img.jpg'), _roi)
      found_left, found_right = self.get_near_contour(img, file_name, contours, i, parameters)
      for j, __contours in enumerate([found_left, found_right]):
        for z, _contour in enumerate(__contours):
          _x, _y, _w, _h = cv2.boundingRect(_contour)
          if self.debug:
            _roi = img[_y:_y + _h, _x:_x + _w]
            name = file_name + '_' + '00000' + str(i)[-5:]
            filedir = os.path.join(STORAGE_BASE, REPOSITORY)
            filedir += '_' + self.newspaper.name.lower().replace(' ', '_')
            Path(filedir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(filedir, name + '_' + str(j) + '_' + str(z) + '.jpg'), _roi)
      flag = False
      count = 0
      valid_contours = []
      valid_contours.append(contour)
      # controlla se i caratteri a sinistra hanno una distanza fra di loro di meno di parameters.left_free[1]
      for j, _contour in enumerate(found_left):
        _x, _y, _w, _h = cv2.boundingRect(_contour)
        if j < parameters.max_near - 1 and x - _x - _w < max(_w, w, parameters.left_free[1]) // 3:
          if _w > _h * 1.1 or _x == 0 or _x + _w >= width:
            flag = True
            break
          x = _x
          count += 1
          valid_contours.append(_contour)
          continue
        if j >= parameters.max_near - 1 and x - _x - _w < max(_w, w, parameters.left_free[1]) // 3:
          flag = True
          break
      # controlla se c'è uno spazio vuoto lungo almeno parameters.left_free[0] a sinistra dell'ultimo carattere valido
      if not flag and count < len(found_left):
        _x, _y, _w, _h = cv2.boundingRect(found_left[count])
        if x - _x - _w < parameters.left_free[0]:
          flag = True
      # controlla se i caratteri a destra hanno una distanza fra di loro di meno di parameters.right_free[1]
      x, y, w, h = cv2.boundingRect(contour)
      _count = 0
      if not flag:
        for j, _contour in enumerate(found_right):
          _x, _y, _w, _h = cv2.boundingRect(_contour)
          if j < parameters.max_near - 1 and _x - x - w < max(_w, w, parameters.right_free[1]) // 3:
            if _w > _h * 1.1 or _x == 0 or _x + _w >= width:
              flag = True
              break
            x = _x
            w = _w
            count += 1
            _count += 1
            valid_contours.append(_contour)
            if count > parameters.max_near - 1 :
              flag = True
              break
            continue
          if j >=  parameters.max_near - 1 and _x - x - w < max(_w, w, parameters.right_free[1]) // 3:
            flag = True
            break
      # controlla se c'è uno spazio vuoto lungo almeno parameters.right_free[0] a destra dell'ultimo carattere valido
      if not flag and _count < len(found_right):
        _x, _y, _w, _h = cv2.boundingRect(found_right[_count])
        if _x - x - w < parameters.right_free[0]:
          flag = True
      x, y, w, h = cv2.boundingRect(contour)
      if not flag:
        min_w = None
        min_h = None
        # calcola il più piccolo w e h fra i probabili valori numerici
        for i, _contour in enumerate(valid_contours):
          if self.debug:
            _x, _y, _w, _h = cv2.boundingRect(_contour)
            _roi = img[_y:_y + _h, _x:_x + _w]
            name = file_name + '_' + '00000' + str(i)[-5:]
            filedir = os.path.join(STORAGE_BASE, REPOSITORY)
            filedir += '_' + self.newspaper.name.lower().replace(' ', '_')
            Path(filedir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(os.path.join(filedir, name + '_valid.jpg'), _roi)
          if min_w is None:
            min_w = cv2.boundingRect(_contour)[2]
            min_h = cv2.boundingRect(_contour)[3]
            continue
          min_w = min(min_w, cv2.boundingRect(_contour)[2])
          min_h = min(min_h, cv2.boundingRect(_contour)[3])
        # controlla che non vi siano contour sopra (o sotto) escludendo dal controllo i poligoni piccoli
        for i, _contour in enumerate(contours):
          _x, _y, _w, _h = cv2.boundingRect(_contour)
          if _w >= parameters.box[1] or _h >= parameters.box[3] or _w > _h * 1.25 or _w <= min_w // 2 or _h <= min_h // 2:
            continue
          # if (parameters.position == 'top' and _x > x - w * 10 and _x < x + w * 10 and _y < y - h // 2) \
          #     or (parameters.position == 'bottom' and _x > x - w * 10 and _x < x + w * 10 and _y > y + h // 2):
          if (parameters.position == 'top' and _y < y - h // 2) or (parameters.position == 'bottom' and _y > y + h // 2):
            flag = True
      if not flag:
        _contours.append(contour)
    return _contours
  def add_boxes(self, images, img, contours, hierarchy, parameters, file_name, no_resize, part):
    def _get_contours(e):
      return e[0]
    _contours = []
    _check = []

    if parameters.right_free is not None:
      contours = self.clean_contours(img, file_name, contours, hierarchy, parameters)
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour) # hierarchy[0, i, 3] == -1 and
      if (parameters.can_be_internal or hierarchy[0, i, 3] == -1) and w > parameters.box[0] and w < parameters.box[1] and h > parameters.box[2] and h < parameters.box[3]:
        roi = img[y:y + h, x:x + w]
        mean = roi.mean()
        if mean >= parameters.min_mean and mean <= parameters.max_mean:
          # if (parameters.can_be_internal and not (x, y, w, h) in _check) or hierarchy[0, i, 3] == -1:
          add = True
          if parameters.max_distance is not None:
            for _x, _y in _check:
              if abs(x - _x) <= parameters.max_distance and abs(y - _y) <= parameters.max_distance:
                add = False
                break
          if add:
            _contours.append((x, y, contour))
            _check.append((x, y))
    _contours.sort(key = _get_contours)
    for i, (_, _, contour) in enumerate(_contours):
      x, y, w, h = cv2.boundingRect(contour)
      perimeter = cv2.arcLength(contour, True)
      roi = img[y:y + h, x:x + w]
      name = file_name + '_' + str(part) + '_' + ('00000' + str(i + 1))[-5:]
      if not no_resize:
        roi = cv2.resize(roi, NUMBER_IMAGE_SIZE)
      if parameters.internal_box is not None:
        cnts_inside, hierarchy_inside = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _cnts_inside = []
        for i, cnt_inside in enumerate(cnts_inside):
          _x, _, _, _ = cv2.boundingRect(cnt_inside)
          _cnts_inside.append((_x, cnt_inside))
        _cnts_inside.sort(key=_get_contours)
        for j, (_, cnt_inside) in enumerate(_cnts_inside):
          _x, _y, _w, _h = cv2.boundingRect(cnt_inside)
          _roi = roi[_y:_y + _h, _x:_x + _w]
          name_i = file_name + '_' + str(part) + '_' + ('00000' + str(j + 1))[-5:]
          if (_w > parameters.internal_box[0] and
             _w < parameters.internal_box[1] and
             _h > parameters.internal_box[2] and
             _h < parameters.internal_box[3]):
            if not no_resize:
              _roi = cv2.resize(_roi, NUMBER_IMAGE_SIZE)
            images.append((name_i, _roi, perimeter))
      else:
        images.append((name, roi, perimeter))
  def get_boxes(self, image, parameters, level=200, no_resize=False, part=0):
    def _ordering_contours(c):
      x, y, w, h = cv2.boundingRect(c)
      return (x, w)
    def isgray(img):
      if len(img.shape) < 3:
        return True
      if img.shape[2] == 1:
        return True
      r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
      if np.allclose(r, g) and np.allclose(r, b):
        return True
      return False
    img = self.change_contrast_PIL(image, level)
    if isgray(img):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = self.change_contrast_cv2(img)
    if parameters is None:
      parameters = self.newspaper.get_parameters()
      if isinstance(parameters, list):
        parameters = parameters[0]
    p = self.file_name.split('_')[-1][1:]
    if parameters.include is not None:
      if not int(p) in parameters.include:
        return None, img
    if parameters.exclude is not None and p.isdigit() and int(p) in parameters.exclude:
        return None, img
    img = self.cv2_resize(img, parameters.scale)
    heigth, weight, _ = img.shape
    # bimg = img.copy()
    # Questo cancella tutto ciò che non è
    if parameters.exclude_colors is not None and len(parameters.exclude_colors) == 3:
      if parameters.exclude_colors[0] >= 0:
        red = np.where(img[:, :, 0] > parameters.exclude_colors[0])
      else:
        red = np.where(img[:, :, 0] < -parameters.exclude_colors[0])
      if parameters.exclude_colors[1] >= 0:
        green = np.where(img[:, :, 1] > parameters.exclude_colors[1])
      else:
        green = np.where(img[:, :, 1] < -parameters.exclude_colors[1])
      if parameters.exclude_colors[2] >= 0:
        blue = np.where(img[:, :, 2] > parameters.exclude_colors[2])
      else:
        blue = np.where(img[:, :, 2] < -parameters.exclude_colors[2])
      black_pixels = (np.concatenate((red[0], green[0], blue[0]), axis = 0), np.concatenate((red[1], green[1], blue[1]), axis = 0))
      img[black_pixels] = [255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bimg = img.copy()
    # Questo riempie i buchi
    if parameters.fill_hole is not None:
      if parameters.invert_fill_hole:
        thresh, binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      else:
        thresh, binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      kernel = np.ones((parameters.fill_hole, parameters.fill_hole), np.uint8)
      # binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=5)
      gray = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
      if parameters.invert_fill_hole:
        gray = 255 - gray
    # Remove horizontal lines
    if parameters.delete_horizontal:
      horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
      remove_horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
      cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      for c in cnts:
        cv2.drawContours(gray, [c], -1, (255, 255, 255), 5)
    # Remove vertical lines
    if parameters.delete_vertical:
      vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
      remove_vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
      cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      for c in cnts:
        cv2.drawContours(gray, [c], -1, (255,255,255), 5)
    img = cv2.threshold(gray, parameters.ts, 255, cv2.THRESH_BINARY_INV)[1]
    if not parameters.invert:
      img = 255 - img
    bimg = img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(img, 1, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=_ordering_contours)
    # cancella i contorni con area < 100 e > 25000 o spessi meno di 10
    _contours = []
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      # if w * h > parameters.min_area and w * h < 25000 and x > 0 and y > 0 and x + w < weight and y + h < heigth:
      if (parameters.min_area is None or w * h > parameters.min_area) and w * h < 25000 and w >= parameters.less_w / 2 and h >= parameters.less_h / 2:
        _contours.append(contour)
    contours = _contours
    # cancella i contorni doppi
    _contours = []
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      flag = False
      for j, _contour in enumerate(contours):
        if j >= i:
          break
        _x, _y, _w, _h = cv2.boundingRect(_contour)
        if x == _x and y == _y and w == _w and h == _h:
          flag = True
          break
      if not flag:
        _contours.append(contour)
    contours = _contours
    _contours = []
    # cancella i contorni interni
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      flag = False
      for j, _contour in enumerate(contours):
        if j == i:
          continue
        _x, _y, _w, _h = cv2.boundingRect(_contour)
        # if _w <= parameters.box[0] or _w >= parameters.box[1] or _h <= parameters.box[2] or _h >= parameters.box[3]:
        #   continue
        if (x >= _x and x + w <= _x + _w and y >= _y and y + h <= _y + _h):
          flag = True
          break
      if not flag:
        _contours.append(contour)
    contours = _contours
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(bimg, (x, y), (x + w, y + h), (36, 255, 12), 2)
    file_name = Path(self.file_name).stem
    images = [(self.file_name + '_' + str(part) + '_00000', bimg)]
    self.add_boxes(images, img, contours, hierarchy, parameters, file_name, no_resize, part)
    return images, img
  def get_pages_numbers(self, no_resize = False, filedir = None, save_head = True, force=False):
    if self.newspaper.isfirstpage:
      return None
    if self.isAlreadySeen() or force:
      images = self.get_images_list(no_resize = no_resize)
      if images is not None and not save_head and len(images):
        images.pop(0)
      if filedir is not None and images is not None:
        for img in images:
          image = Image.fromarray(img[1])
          Path(filedir).mkdir(parents=True, exist_ok=True)
          file_name = img[0]
          if 'INS' in self.original_path:
            dir = self.original_path.split('/')[-1]
            n = dir.split(' ')[2]
            file_name = '_'.join(file_name.split('_')[:-3]) + ('_INS_' + str(n)) + '_' + '_'.join(file_name.split('_')[-3:])
          image.save(os.path.join(filedir, file_name + '.jpg'), format="JPEG")
        images = None
      return images
    return None
  def get_crop(self):
    image = Image.open(self.original_image)
    cropped = self.newspaper.crop_png(image)
    Path(self.filedir).mkdir(parents=True, exist_ok=True)
    if not self.no_resize:
      cropped = cropped.resize(NUMBER_IMAGE_SIZE, Image.Resampling.LANCZOS)
    cropped.save(os.path.join(self.filedir, self.file_name + '.jpg'), format="JPEG", quality=20)
    return image
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
        f[l+12] == '_'
  def extract_page(self):
    return self.newspaper.get_page()
  def add_pdf_metadata(self, first_number = None):
    if not os.path.isfile(self.pdf_file_name):
      return False
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
        '/Producer': 'osi-servizi-informatici@cloud - Milano'
      })
      file_out = open(self.pdf_file_name, 'wb')
      pdf_merger.write(file_out)
      file_in.close()
      file_out.close()
      os.remove(self.pdf_file_name + '.2')
    except Exception as e:
      os.rename(self.pdf_file_name + '.2', self.pdf_file_name)
      # file_in.write(self.pdf_file_name)
      return 0
    return 1

  def get_jpg_metadata(self, image, first_number=None):
    exif = image.getexif()
    exif[0x9286] = \
      'Nome del periodico:' + self.newspaper.name \
      + ' ; Anno:' + str(self.year) \
      + ' ; Mese:' + str(self.month) \
      + ' ; Giorno:' + str(self.day) \
      + ' ; Numero del quotidiano:' + str(int(self.newspaper.number) + first_number) \
      + ' ; Anno del quotidiano:' + self.newspaper.year \
      + ' ; Nome_del_periodico:' + self.newspaper.name \
      + ' ; Anno:' + str(self.year) \
      + ' ; Mese:' + str(self.month) \
      + ' ; Giorno:' + str(self.day) \
      + ' ; Data:' + str(self.newspaper.date) \
      + ' ; Pagina:' + str(self.newspaper.n_page) \
      + ' ; Numero_del_quotidiano:' + str(self.newspaper.number) \
      + ' ; Anno_del_quotidiano:' + str(self.newspaper.year)
    return exif
  def add_jpg_metadata(self, image, first_number = None):
    if not os.path.isfile(self.pdf_file_name):
      return
    if first_number is None:
      first_number = 0
    if self.isAlreadySeen():
      if os.path.isdir(self.pdf_path):
        filedir, dirs, files = next(os.walk(self.pdf_path))
        for dir in dirs:
          if dir != 'pdf':
            image = Image.open(os.path.join(filedir, dir, self.original_file_name + '.tif'))
            exif = self.get_jpg_metadata(image, first_number)
            image.save(os.path.join(STORAGE_BASE, 'tmp', self.original_file_name + '_' + dir + '.tif'), exif=exif)
            # image.save(os.path.join(filedir, dir, self.original_file_name + '.jpg'), exif=exif)
            pass
  def get_prediction(self, no_resize=False):
    images = self.get_images_list(no_resize = no_resize)
    if images is not None and len(images):
      head_image, prediction, predictions = self.get_page_numbers(self.model, images)
      return prediction
    return None
  def get_images_list(self, no_resize=False):
    image = Image.open(self.original_image)
    if not self.isins:
      cropped = self.newspaper.crop_png(image)
      parameters = self.newspaper.get_parameters()
    else:
      cropped = self.newspaper.crop_ins_png(image)
      parameters = self.newspaper.get_ins_parameters()
    if isinstance(cropped, list):
      images = []
      for i in range(len(cropped)):
        position = self.newspaper.get_page_ins_position()[i]
        if isinstance(parameters, list):
          parameters[i].position = position
          _images, _ = self.get_boxes(cropped[i], no_resize=no_resize, parameters=parameters[i], part=i + 1)
        else:
          parameters.position = position
          _images, _ = self.get_boxes(cropped[i], no_resize=no_resize, parameters=parameters, part=i + 1)
        if _images is None:
          continue
        images.insert(i, _images[0])
        for j in range(1, len(_images)):
          images.append(_images[j])
    else:
      position = self.newspaper.get_page_ins_position()
      parameters.position = position
      images, _ = self.get_boxes(cropped, parameters=parameters, no_resize=no_resize)
    return images

  def check_pages_numbers(self, model, no_resize=False):
    if self.isAlreadySeen():
      if str(self.newspaper.n_page)[0] == '?':
        self.page_control = 2
        return None, None, None, None, True
      images = self.get_images_list(no_resize = no_resize)
      head_image = None
      prediction = None
      predictions = None
      if images is not None and len(images):
        head_image, prediction, predictions = self.get_page_numbers(model, images)
      if images is None or prediction is None:
        self.page_control = -1
      elif prediction == self.newspaper.n_page:
        self.page_control = 1
      else:
        self.page_control = 0
      return head_image, images, prediction, predictions, True
    return None, None, None, None, False
  def get_page_numbers(self, model, images):
    def isgray(img):
      if len(img.shape) < 3:
        return True
      if img.shape[2] == 1:
        return True
      r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
      if np.allclose(r, g) and np.allclose(r, b):
        return True
      return False

    head_images = []
    i = 0
    while i < len(images):
      name = images[i][0]
      if name.split('_')[-1] == '00000':
        hi = images.pop(i)
        title = hi[0]
        # head_image = cv2.cvtColor(head_image[1], cv2.COLOR_GRAY2RGB)
        head_image = Image.fromarray(hi[1])
        head_image = tf.image.convert_image_dtype(head_image, dtype=tf.float32)
        head_images.append((title, head_image))
      else:
        i += 1
    dataset = []
    for image in images:
      img = cv2.cvtColor(image[1], cv2.COLOR_GRAY2RGB)
      img = Image.fromarray(img)
      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      dataset.append(img)
    try:
      original_predictions = list(np.argmax(model.predict(np.array(dataset), verbose = 0), axis=-1))
    except Exception as e:
      return None, None, None
    b = None
    predictions = []
    last = len(self.newspaper.get_dictionary()) - 1
    for e in original_predictions:
      if b is None and (e == 0 or e == last):
        continue
      if e != last:
        predictions.append(str(e))
      b = e
    _predictions = [self.newspaper.get_dictionary()[int(p)] for p in predictions]
    prefix = self.newspaper.get_prefix()
    init = ''.join(_predictions).find(prefix)
    if prefix != '' and init >= 0:
      _predictions = _predictions[init + len(prefix) : ]
    if len(_predictions) and ''.join(_predictions).isdigit():
      prediction = int(''.join(_predictions))
    else:
      prediction = None
    return head_images, prediction, original_predictions
  def open_win_pages_files(self, image, file_to_be_changing, n_unkown, prediction = None):
    def close():
      gui.destroy()
      exit()
    def end():
      global end_flag
      gui.destroy()
      end_flag = True
    def number_chosen(button_press):
      global new_file
      global end_flag
      global flag_digited
      global _n_unkown
      end_flag = False
      if button_press == '?':
        _n_unkown += 1
        on = self.file_name.split('_')[-1][1:]
        button_press = '?' + ('00000' + str(_n_unkown))[-len(on) + 1:]
      if button_press != 'ok':
        new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p' + str(button_press)
        label1.config(text=new_file)
        if self.file_name != os.path.basename(new_file) and new_file[-1] != 'p':
          on = self.file_name.split('_')[-1][1:]
          n = ('00000' + new_file.split('_')[-1][1:])[-len(on):]
          new_file = '_'.join(new_file.split('_')[:-1]) + '_p' + n
          ext = pathlib.Path(self.original_image).suffix
          file_to_be_changing.append((os.path.join(self.original_path, self.original_image), os.path.join(self.original_path, new_file) + ext))
          if os.path.isdir(self.pdf_path):
            for filedir, _, files in os.walk(self.pdf_path):
              for file in files:
                ext = pathlib.Path(file).suffix
                if Path(file).stem == self.file_name:
                  file_to_be_changing.append((os.path.join(filedir, file), os.path.join(filedir, new_file) + ext))
      gui.destroy()
    def page_chosen(button_press):
      global next_page
      next_page = int(button_press)
      gui.destroy()
    def on_start_hover(event):
      global label3
      img = image.crop((event.x - 40, event.y - 40, event.x + 40, event.y + 40))
      img = img.resize((480, 480), Image.Resampling.LANCZOS)
      crop = ImageTk.PhotoImage(img)
      label3.configure(image=crop)
      label3.image = crop
    global new_file
    global end_flag
    global next_page
    global label3
    global _n_unkown
    _n_unkown = n_unkown
    end_flag = False
    next_page = -1
    gui = tk.Tk()
    gui.title('ATTENZIONE ! Se confermi verrà mdificato il nome del file in tutti i formati esistenti: ' + self.file_name)
    w = 2360  # Width
    h = 2200  # Height
    TOTAL_BUTTON_IN_LINE = 10
    screen_width = gui.winfo_screenwidth()
    screen_height = gui.winfo_screenheight()
    x = (screen_width / 2) - (w / 2)
    y = (screen_height / 2) - (h / 2)
    gui.geometry('%dx%d+%d+%d' % (w, h, x, y))
    gui_frame = tk.Frame(gui)
    gui_frame.pack(fill=tk.X, side=tk.BOTTOM)
    # new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p'
    new_file = self.file_name
    if prediction is not None:
      new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p' + str(prediction)
    label1 = Label(gui, text=new_file, font=('Arial 20 bold'), height = 5)
    label1.pack(padx=(10, 10), pady=(10, 10))
    image = image.resize((1550, 2000), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(image)
    label2 = Label(gui_frame, image=img)
    label2.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    button_frame = tk.Frame(gui_frame)
    button_frame.grid(row=0, column=1, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    if self.newspaper.n_page % 2 == 0:
      crop = ImageTk.PhotoImage(image.crop((0, 0, 240, 240)).resize((480, 480), Image.Resampling.LANCZOS))
    else:
      w, h = image.size
      crop = ImageTk.PhotoImage(image.crop((w - 240, 0, w, 240)).resize((480, 480), Image.Resampling.LANCZOS))
    label3 = Label(button_frame, image=crop)
    label3.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    label2.bind('<Button-1>', on_start_hover)
    button_frame_1 = tk.Frame(button_frame)
    button_frame_1.grid(row=1, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    n_lines = (self.newspaper.n_pages + 2) // TOTAL_BUTTON_IN_LINE + 2
    buttons = [[0 for x in range(TOTAL_BUTTON_IN_LINE)] for x in range(n_lines)]
    for i in range(n_lines):
      for j in range(TOTAL_BUTTON_IN_LINE):
        text = i * TOTAL_BUTTON_IN_LINE + j + 1
        if text == self.newspaper.n_pages + 1 + 8:
          text = 'ok'
        elif text == self.newspaper.n_pages + 2 + 8:
          text = '?'
        elif text > self.newspaper.n_pages + 8:
          break
        pixel = tk.PhotoImage(width=1, height=1)
        buttons[i][j] = tk.Button(button_frame_1,
                                  text=text,
                                  compound="center",
                                  font=('Aria', 14),
                                  height=2,
                                  width=4,
                                  padx=0,
                                  pady=0,
                                  command=lambda number=str(text): number_chosen(number))
        buttons[i][j].grid(row=i, column=j, sticky=tk.W + tk.E, padx=(2, 2), pady=(2, 2))
    button_frame_2 = tk.Frame(button_frame)
    button_frame_2.grid(row=2, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    n_lines = self.newspaper.n_pages // TOTAL_BUTTON_IN_LINE + 1
    buttons = [[0 for x in range(TOTAL_BUTTON_IN_LINE)] for x in range(n_lines)]
    lst = [x + 1 for x in range(self.newspaper.n_pages)]
    for old_file, new_file in file_to_be_changing:
      if pathlib.Path(old_file).suffix == '.tif':
        on = ''.join(Path(old_file).stem.split('_')[-1])[1:]
        nn = ''.join(Path(new_file).stem.split('_')[-1])[1:]
        if on.isdigit():
          on = int(on)
        if nn.isdigit():
          nn = int(nn)
        if on in lst:
          lst.remove(on)
        lst.append(nn)
        pass
    for i in range(n_lines):
      for j in range(TOTAL_BUTTON_IN_LINE):
        text = i * TOTAL_BUTTON_IN_LINE + j + 1
        if text > self.newspaper.n_pages:
          break
        pixel = tk.PhotoImage(width=1, height=1)
        buttons[i][j] = tk.Button(button_frame_2,
                                  text=text,
                                  compound="center",
                                  font=('Aria', 14),
                                  height=2,
                                  width=4,
                                  padx=0,
                                  pady=0,
                                  command=lambda number=str(text): page_chosen(number))
        if not text in lst:
          buttons[i][j].config(bg='#f00', fg='#fff')
        buttons[i][j].grid(row=i, column=j, sticky=tk.W + tk.E, padx=(2, 2), pady=(2, 2))
    end_button = Button(button_frame, text="Fine", font=('Arial', 18), command=end, height=2, width=4)
    end_button.grid(row=3, column=0, sticky=tk.W + tk.E, padx=(5, 5), pady=(5, 5))
    exit_button = Button(button_frame, text="Esci", font=('Arial', 18), command=close, height=2, width=4)
    exit_button.grid(row=4, column=0, sticky=tk.W + tk.E, padx=(5, 5), pady=(5, 5))
    gui.mainloop()
    return file_to_be_changing, end_flag, next_page, _n_unkown
  def rename_pages_files(self, file_to_be_changing, n_unkown, model = None):
    if self.isAlreadySeen():
      if os.path.isdir(self.pdf_path):
        filedir, dirs, files = next(os.walk(self.pdf_path))
        for dir in dirs:
          if dir != 'pdf':
            image = Image.open(os.path.join(filedir, dir, self.original_file_name + '.jpg'))
            break
      else:
        image = Image.open(self.original_image)
      prediction = None
      if model is not None:
        _, _, prediction, _ = self.check_pages_numbers(model)
      file_to_be_changing, end_flag, next_page, n_unkown = self.open_win_pages_files(image, file_to_be_changing, n_unkown, prediction = prediction)
    return file_to_be_changing, end_flag, next_page, n_unkown
  def convert_image(self, force):
    image = Image.open(self.original_image)
    for convert in self.conversions:
      self.convert_image_single_conversion(convert, image, force)

  def convert_image_single_conversion(self, convert, image, force):
    path_image = os.path.join(self.jpg_path, convert.image_path)
    Path(path_image).mkdir(parents=True, exist_ok=True)
    file = os.path.join(path_image, self.file_name) + '.jpg'
    if force or not Path(file).is_file():
      if image.size[0] < image.size[1]:
        wpercent = (convert.resolution / float(image.size[1]))
        xsize = int((float(image.size[0]) * float(wpercent)))
        image = image.resize((xsize, convert.resolution), Image.Resampling.LANCZOS)
      else:
        wpercent = (convert.resolution / float(image.size[0]))
        ysize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((convert.resolution, ysize), Image.Resampling.LANCZOS)
      image.save(file, 'JPEG', dpi=(convert.dpi, convert.dpi), quality=convert.quality)

  def set_bobine_select_images(self):
    def _order(e):
      return e[0]

    count_n = 1
    count = 0
    file = self.original_image
    page_n = '00' + str(self.newspaper.n_page) if self.newspaper.n_page < 10 else '0' + str(
      self.newspaper.n_page) if self.newspaper.n_page < 100 else str(self.newspaper.n_page)
    if Path(self.original_image).stem[:5] != 'merge':
      return
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if self.threshold is not None:
      ret, thresh = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
    else:
      thresh = img.copy()
    # Questo riempie i buchi bianchi
    fill_hole = 8
    invert_fill_hole = False
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    # Questo riempie i buchi neri
    fill_hole = 32
    invert_fill_hole = True
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bimg = img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    books = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      rect = cv2.minAreaRect(contour)
      if self.debug:
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(bimg, [box], 0, (0, 255, 0), 3)
      if w > 10000 and h > 5000 and w < 15000:
        books.append((x, y, w, h))
    books.sort(key=_order)
    for book in books:
      x = book[0]
      y = book[1]
      w = book[2]
      h = book[3]
      _, _, weight, _ = cv2.boundingRect(img)
      if x != 0 and x + w != weight:
        cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 5)
    if self.debug:
      n = '00' + str(count_n) if count_n < 10 else '0' + str(count_n) if count_n < 100 else str(count_n)
      file_bimg = os.path.join(self.filedir, 'fotogrammi_bing_' + page_n + '_' + n + '.tif')
      file_thresh = os.path.join(self.filedir, 'fotogrammi_thresh_' + page_n + '_' + n + '.tif')
      cv2.imwrite(file_bimg, bimg)
      cv2.imwrite(file_thresh, thresh)
      count_n += 1
    global_file_list = []
    for book in books:
      x = book[0]
      y = book[1]
      w = book[2]
      h = book[3]
      n = '00' + str(count_n) if count_n < 10 else '0' + str(count_n) if count_n < 100 else str(count_n)
      file2 = os.path.join(self.filedir, 'fotogrammi_' + page_n + '_' + n + '.tif')
      _, _, weight, _ = cv2.boundingRect(img)
      if x != 0 and x + w != weight:
        roi = img[y:y + h, x:x + w]
        cv2.imwrite(file2, roi)
        global_file_list.append((file2, x, y, w, h))
        count_n += 1
        count += 1
    if self.remove_merge:
      os.remove(file)
    return (global_file_list, count)

  def rotate_frames(self):
    def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    count = 0
    file = self.original_image
    file_bing = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_ning = '.'.join(file.split('.')[:-1]) + '_ning.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if self.angle is not None:
      img = rotate_image(img, self.angle)
      cv2.imwrite(file, img)
      return 1
    if self.threshold is not None:
      ret, thresh = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
    else:
      thresh = img.copy()
    # Questo riempie i buchi bianchi
    fill_hole = 8
    invert_fill_hole = False
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    # Questo riempie i buchi neri
    fill_hole = 32
    invert_fill_hole = True
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bimg = img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    books = []
    rect = None
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if w > self.limit and h > self.limit:
        books.append(contour)
    if len(books) > 0:
      contour = books[0]
      x, y, w, h = cv2.boundingRect(contour)
      rect = cv2.minAreaRect(contour)
      if DEBUG:
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(bimg, [box], 0, (0, 255, 0), 3)
    if rect is not None:
      angle = rect[2]
      if angle < 45:
        angle = 90 + angle
      if angle > 85 and (angle < 89.9 or angle > 90.1):
        angle = angle - 90
        if angle < 5.0:
          img = rotate_image(img, angle)
        count += 1
      if DEBUG:
        cv2.imwrite(file_ning, img)
        cv2.imwrite(file_bing, bimg)
        cv2.imwrite(file_thresh, thresh)
      else:
        cv2.imwrite(file, img)
    return count
  def divide_image(self):
    file_name_no_ext = Path(self.original_image).stem
    file_path_no_ext = os.path.join(self.filedir, file_name_no_ext)
    ext = Path(self.original_image).suffix
    # img = cv2.imread(self.original_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(self.original_image)
    imgs = self.newspaper.divide(img)
    for i, _img in enumerate(imgs):
      cv2.imwrite(file_path_no_ext + '_' + str(i + 1) + ext, _img)
    os.remove(self.original_image)
    self.isdivided = True
    return 1
  def drawline(self, img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
      r = i / dist
      x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
      y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
      p = (x, y)
      pts.append(p)
    if style == 'dotted':
      for p in pts:
        cv2.circle(img, p, thickness, color, -1)
    else:
      s = pts[0]
      e = pts[0]
      i = 0
      for p in pts:
        s = e
        e = p
        if i % 2 == 1:
          cv2.line(img, s, e, color, thickness)
        i += 1
  def drawpoly(self, img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
      s = e
      e = p
      self.drawline(img, s, e, color, thickness, style)

  def drawrect(self, img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    self.drawpoly(img, pts, color, thickness, style)
  def remove_all_lines(self, img):
    img = self.remove_horizontal_lines(img, 1000, 10)
    img = self.remove_vertical_lines(img, 10, 1000)
    img = self.remove_horizontal_lines(img, 500, 10)
    img = self.remove_vertical_lines(img, 10, 500)
    img = self.remove_horizontal_lines(img, 100, 10)
    img = self.remove_vertical_lines(img, 10, 100)
    return img
  def clean_images(self):
    def union(a, b):
      x = min(a[0], b[0])
      y = min(a[1], b[1])
      w = max(a[0] + a[2], b[0] + b[2]) - x
      h = max(a[1] + a[3], b[1] + b[3]) - y
      return (x, y, w, h)
    def check_intersection(l1, r1, l2, r2):
      # if rectangle has area 0, no overlap
      if l1[0] == r1[0] or l1[1] == r1[1] or r2[0] == l2[0] or l2[1] == r2[1]:
        return False
      # If one rectangle is on left side of other
      if l1[0] > r2[0] or l2[0] > r1[0]:
        return False
      # If one rectangle is above other
      if r1[1] > l2[1] or r2[1] > l1[1]:
        return False
      return True
    def checkIntersection(boxA, boxB):
      x = max(boxA[0], boxB[0])
      y = max(boxA[1], boxB[1])
      w = min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - x
      h = min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - y
      foundIntersect = True
      if w < 0 or h < 0:
        foundIntersect = False
      return foundIntersect
    def union(contours, always_merge = False):
      j = 0
      while j < len(contours) - 1:
        contour = contours[j]
        x, y, w, h = cv2.boundingRect(contour)
        flag = False
        for i, _contour in enumerate(contours):
          if i == j:
            continue
          _x, _y, _w, _h = cv2.boundingRect(_contour)
          if _x >= x and _y >= y and _x + _w <= x + w and _y + _h <= y + h:
            del contours[i]
            flag = True
            break
          if always_merge:
            merge = True
          else:
            merge = checkIntersection((x, y, w, h), (_x, _y, _w, _h))
          if merge:
            points = np.vstack([contour, _contour])
            ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            ctr = cv2.convexHull(ctr)
            x, y, w, h = cv2.boundingRect(ctr)
            contour = np.array([(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1)]).reshape(
              (-1, 1, 2)).astype(np.int32)
            contours[j] = contour
            del contours[i]
            flag = True
            break
        if not flag:
          j += 1
      return contours
    def _ordering_contours(c):
      x, y, w, h = cv2.boundingRect(c)
      return (x, w)
    count = 0
    file = self.original_image
    file_nimg = '.'.join(file.split('.')[:-1]) + '_ning.' + file.split('.')[-1]
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_mem = '.'.join(file.split('.')[:-1]) + '_mem.' + file.split('.')[-1]
    file_dimg = '.'.join(file.split('.')[:-1]) + '_ding.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    _img = img.copy()
    img = cv2.convertScaleAbs(img, alpha=1.01, beta=0)
    oh, ow = img.shape
    ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    # fill contours with white
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      if w < 100 or h < 100:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    # ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    #get new contours, enlarge them and put in order
    bimg = _img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    thresh = self.remove_all_lines(thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      # if DEBUG:
      #   cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 3)
    lx, ly, x_ofset, y_ofset = self.newspaper.get_limits()
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      # cv2.rectangle(bimg, (x, y), (x + w, y + h), (255, 0, 0), 3)
      if x > 0 and y > 0 and w > 50 and h > 50 and w < lx * 0.8 and h < ly * 0.8:
        p = []
        ofset = 10 if w > 300 else 20
        x = x - ofset if ofset < x else 0
        y = y - ofset if ofset < y else 0
        w += ofset * 2
        w = w if x + w < ow else ow - x - 1
        h += ofset * 2
        h = h if y + h < oh else oh - y - 1
        # if x < ow // 2 and x + w > ow // 2:
        #   continue
        cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 16)
        p.append((x, y))
        p.append((x + w, y))
        p.append((x + w, y + h))
        p.append((x, y + h))
        ctr = np.array(np.array(p)).reshape((-1, 1, 2)).astype(np.int32)
        cnts.append(ctr)
    contours = list(cnts)
    contours.sort(key=_ordering_contours)
    # for contour in contours:
    #   x, y, w, h = cv2.boundingRect(contour)
    #   if DEBUG:
    #     cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # join all the overlapping contours
    contours = union(contours)
    # fill contours in black
    nimg = _img.copy()
    white_nimg = 255 - np.zeros_like(img)
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if DEBUG:
        cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), 0, -1)
      cv2.rectangle(nimg, (x, y), (x + w, y + h), 0, -1)
      cv2.rectangle(white_nimg, (x, y), (x + w, y + h), 0, -1)
    _contours = union(contours, True)
    dimg = nimg.copy()
    # dimg = cv2.convertScaleAbs(dimg, alpha=1.1, beta=5)
    threshold = self.threshold
    df_describe = pd.DataFrame(dimg[white_nimg > 0])
    _threshold = df_describe.describe(percentiles=[0.2]).at['20%', 0]
    threshold = _threshold if _threshold < threshold else threshold
    threshold = threshold if threshold < 225 else 225
    self.final_threshold = self.final_threshold if threshold >= 180 else self.final_threshold - (180 - threshold) // 2
    threshold = threshold if threshold > self.final_threshold else self.final_threshold
    threshold = threshold if threshold >= 160 else 160
    dimg[dimg >= threshold] = self.color
    dimg[dimg < threshold] = dimg[dimg < threshold] * 0.96
    # dimg[(dimg < 150) & (dimg >= 50)] = dimg[(dimg < 150) & (dimg >= 50)] - 48
    # dimg[(dimg < 180) & (dimg >= 150)] = dimg[(dimg < 180) & (dimg >= 150)] - 32
    # dimg[(dimg < 200) & (dimg >= 180)] = dimg[(dimg < 200) & (dimg >= 180)] - 24
    # dimg[(dimg < threshold) & (dimg >= 200)] = dimg[(dimg < threshold) & (dimg >= 200)] - 8
    dimg = cv2.convertScaleAbs(dimg, alpha=0.84, beta=0)
    dimg[dimg >= threshold] = self.color
    dimg[dimg < 24] = 12
    _img = cv2.convertScaleAbs(_img, alpha=1.05, beta=0)
    dimg[white_nimg == 0] = _img[white_nimg == 0]
    dimg[dimg >= 200] = self.color
    for contour in _contours:
      x, y, w, h = cv2.boundingRect(contour)
      if h > ly - y_ofset and y < y_ofset // 2:
        if ly > h:
          _h = ly if ly < oh else oh
          _h = _h if _h > h else h
          y = y - (_h - h) // 2
          y = y if y > 0 else 0
          h = _h
        dimg = dimg[y:y + h, :]
      if w > lx - x_ofset and x < x_ofset // 2:
        if lx > w:
          _w = lx if lx < ow else ow
          _w = _w if _w > w else w
          x = x - (_w - w) // 2
          x = x if x > 0 else 0
          w = _w
        dimg = dimg[:, x:x+w]
      # if DEBUG:
      #   cv2.rectangle(bimg, (x, y), (x + w, y + h), (255, 0, 0), 5)
    if not DEBUG:
      cv2.imwrite(file, dimg)
    else:
      cv2.imwrite(file_bimg, bimg)
      cv2.imwrite(file_nimg, nimg)
      cv2.imwrite(file_dimg, dimg)
      cv2.imwrite(file_mem, white_nimg)
      cv2.imwrite(file_thresh, thresh)
      pass
    return 1
  def remove_borders(self):
    count = 0
    file = self.original_image
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_nimg = '.'.join(file.split('.')[:-1]) + '_ning.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if self.threshold is not None:
      ret, thresh = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
    else:
      thresh = img.copy()
    # Questo riempie i buchi bianchi
    fill_hole = 8
    invert_fill_hole = False
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    # Questo riempie i buchi neri
    fill_hole = 32
    invert_fill_hole = True
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    if invert_fill_hole:
      thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bimg = img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    books = []
    rect = None
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if w > self.limit and h > self.limit:
        books.append(contour)
    if len(books) > 0:
      contour = books[len(books) - 1]
      x, y, w, h = cv2.boundingRect(contour)
      rect = cv2.minAreaRect(contour)
      if DEBUG:
        cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 3)
    lx, ly, x_ofset, y_ofset = self.newspaper.get_limits()
    if rect is not None and w > len(img[1]) // 3 * 2 and h > ly - y_ofset and y < y_ofset // 2:
      img = img[y:y+h, x:x+w]
    else:
      _h, _w = img.shape
      if lx is not None:
        if lx < _w:
          x = (_w - lx) // 2
          img = img[0 : _h, x : x + lx]
        if ly < _h:
          y = (_h - ly) // 2
          img = img[y : y + ly, 0 : _w]
    if DEBUG:
      cv2.imwrite(file_bimg, bimg)
      cv2.imwrite(file_nimg, img)
      cv2.imwrite(file_thresh, thresh)
    else:
      cv2.imwrite(file, img)
    return 1
  def remove_horizontal_lines(self, thresh, x_size=1, y_size=10):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x_size, y_size))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
      cv2.drawContours(thresh, [c], -1, (255, 255, 255), 5)
    return thresh
  def remove_vertical_lines(self, thresh, x_size=1, y_size=10):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x_size, y_size))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
      cv2.drawContours(thresh, [c], -1, (255,255,255), 5)
    return thresh
  def remove_frames(self):
    file = self.original_image
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    file_nimg = '.'.join(file.split('.')[:-1]) + '_nimg.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # if (img[0:6, 0] == np.array([3, 202, 2, 61, 4, 6])).all():
    #   return
    _img = img.copy()
    thresh = img.copy()
    thresh = cv2.convertScaleAbs(thresh, alpha=1.2, beta=0)
    ret, thresh = cv2.threshold(thresh, self.threshold, 255, cv2.THRESH_BINARY)
    # Remove horizontal lines
    thresh = (255 - thresh)
    thresh = self.remove_horizontal_lines(thresh)
    thresh = self.remove_vertical_lines(thresh)
    thresh = (255 - thresh)
    _thresh = thresh.copy()
    h, w = thresh.shape
    max_x_r = 800
    max_x_l = 1000
    max_y = 1000
    limits = [215, 210, 200, 190]
    mx = None
    my = None
    mw = None
    mh = None
    for limit in limits:
      for y in range(self.default_frame[0], max_y):
        mean = max(thresh[y:y+1,0:w // 2].mean(), thresh[y:y+1,w // 2:w].mean())
        # mean = thresh[y:y + 1, 0:w].mean()
        if mean >= limit:
          thresh = thresh[y - 100 if y >= 100 else 0:h, 0:w]
          img = img[y - 100 if y >= 100 else 0:h, 0:w]
          h, w = thresh.shape
          my = y
          mh = h
          break
      if mean >= limit:
        break
    for limit in limits:
      for y in range(h - 1 - self.default_frame[1], h - max_y, -1):
        mean = max(thresh[y:y+1,0:w // 2].mean(), thresh[y:y+1,w // 2:w].mean())
        # mean = thresh[y:y + 1, 0:w].mean()
        if mean >= limit:
          thresh = thresh[0:y + 100, 0:w]
          img = img[0:y + 100, 0:w]
          h, w = thresh.shape
          mh = h
          break
      if mean >= limit:
        break
    for limit in limits:
      for x in range(self.default_frame[2], max_x_l):
        # mean = min(thresh[0:h//2,x:x+1].mean(), thresh[h//2:h,x:x+1].mean())
        mean = thresh[0:h, x:x + 1].mean()
        if mean >= limit:
          thresh = thresh[0: h, x: w]
          img = img[0: h, x: w]
          h, w = thresh.shape
          mx = x
          mw = w
          break
      if mean >= limit:
        break
    for limit in limits:
      for x in range(w - 1 - self.default_frame[3], w - max_x_r, -1):
        # mean = min(thresh[0:h//2,x:x+1].mean(), thresh[h//2:h,x:x+1].mean())
        # mean = max([img[0:_y + h // 4, x:x + 1].mean() for _y in range(0, h, h // 4)])
        mean = thresh[0:h, x:x + 1].mean()
        if mean > 100:
          pass
        if mean >= limit:
          thresh = thresh[0 : h, 0 : x]
          img = img[0 : h, 0 : x]
          h, w = thresh.shape
          mw = w
          break
      if mean >= limit:
        break
    if DEBUG:
      nimg = img.copy()
    h, w = img.shape
    _h, _w = _img.shape
    lx, ly, x_ofset, y_ofset = self.newspaper.get_limits()
    if w < lx - x_ofset or h < ly - y_ofset:
      if _w > lx and w < lx - x_ofset:
        x = (_w - lx) // 2
      else:
        x = mx if mx is not None else 0
        w = mw if mw is not None else lx
      if _h > ly and h < ly - y_ofset:
        y = (_h - ly) // 2
      else:
        y = my if my is not None else 0
        h = mh if mh is not None else ly
      img = _img[y:y + h, x:x + w]
    img[0:6, 0] = [3, 202, 2, 61, 4, 6]
    if DEBUG:
      cv2.imwrite(file_nimg, nimg)
      cv2.imwrite(file_bimg, img)
      cv2.imwrite(file_thresh, _thresh)
    else:
      cv2.imwrite(file, img)
    return 1
  def remove_single_frames(self):
    file = self.original_image
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    file_nimg = '.'.join(file.split('.')[:-1]) + '_nimg.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # if (img[0:6, 0] == np.array([3,202,2,61,4,6])).all():
    #   return
    _img = img.copy()
    thresh = img.copy()
    thresh = cv2.convertScaleAbs(thresh, alpha=1.3, beta=0)
    ret, thresh = cv2.threshold(thresh, self.threshold, 255, cv2.THRESH_BINARY)
    _thresh = thresh.copy()
    h, w = thresh.shape
    max_x_r = 800
    max_x_l = 800
    max_y = 800
    limits = [215, 210, 200, 190]
    mx = None
    my = None
    mw = None
    mh = None
    lx, ly, x_ofset, y_ofset = self.newspaper.get_limits()
    lx = lx // 2
    count = 0
    for limit in limits:
      for y in range(self.default_frame[0], max_y):
        # mean = max(thresh[y:y+1,0:w // 2].mean(), thresh[y:y+1,w // 2:w].mean())
        mean = thresh[y:y + 1, 0:w].mean()
        if mean >= limit:
          # thresh = thresh[y - 100 if y >= 100 else 0:h, 0:w]
          # img = img[y - 100 if y >= 100 else 0:h, 0:w]
          thresh = thresh[y:h, 0:w]
          img = img[y:h, 0:w]
          h, w = thresh.shape
          my = y
          mh = h
          count = 1
          break
      if mean >= limit:
        break
    for limit in limits:
      for y in range(h - 1 - self.default_frame[1], h - max_y, -1):
        # mean = max(thresh[y:y+1,0:w // 2].mean(), thresh[y:y+1,w // 2:w].mean())
        mean = thresh[y:y + 1, 0:w].mean()
        if mean >= limit:
          # thresh = thresh[0:y + 100, 0:w]
          # img = img[0:y + 100, 0:w]
          thresh = thresh[0:y, 0:w]
          img = img[0:y, 0:w]
          h, w = thresh.shape
          mh = h
          count = 1
          break
      if mean >= limit:
        break
    for limit in limits:
      for x in range(self.default_frame[2], max_x_l):
        # mean = min(thresh[0:h//2,x:x+1].mean(), thresh[h//2:h,x:x+1].mean())
        mean = thresh[0:h, x:x + 1].mean()
        if mean >= limit:
          thresh = thresh[0: h, x: w]
          img = img[0: h, x: w]
          h, w = thresh.shape
          mx = x
          mw = w
          count = 1
          break
      if mean >= limit:
        break
    for limit in limits:
      for x in range(w - 1 - self.default_frame[3], w - max_x_r, -1):
        # mean = min(thresh[0:h//2,x:x+1].mean(), thresh[h//2:h,x:x+1].mean())
        mean = thresh[0:h, x:x + 1].mean()
        if mean >= limit:
          img = img[0:h, 0:x]
          h, w = thresh.shape
          mw = w
          count = 1
          break
      if mean >= limit:
        break
    if DEBUG:
      nimg = img.copy()
    h, w = img.shape
    _h, _w = _img.shape
    if w < lx - x_ofset or h < ly - y_ofset:
      if _w > lx and w < lx - x_ofset:
        x = (_w - lx) // 2
      else:
        x = mx if mx is not None else 0
        w = mw if mw is not None else lx
      if _h > ly and h < ly - y_ofset:
        y = (_h - ly) // 2
      else:
        y = my if my is not None else 0
        h = mh if mh is not None else ly
      img = _img[y:y + h, x:x + w]
      count = 1
    img[0:6, 0] = [3, 202, 2, 61, 4, 6]
    if DEBUG:
      cv2.imwrite(file_nimg, nimg)
      cv2.imwrite(file_bimg, img)
      cv2.imwrite(file_thresh, _thresh)
    else:
      cv2.imwrite(file, img)
    return count
  def remove_last_single_frames(self):
    file = self.original_image
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_nimg = '.'.join(file.split('.')[:-1]) + '_nimg.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # if (img[0:6, 0] == np.array([3,202,2,61,4,6])).all():
    #   return
    _img = img.copy()
    h, w = img.shape
    limits = [200]
    for limit in limits:
      for y in range(self.default_frame[0], 0, -1):
        mean = img[y:y + 1, 0:w].mean()
        if mean <= limit:
          img = img[y:h, 0:w]
          h, w = img.shape
          break
      if mean <= limit:
        break
    for limit in limits:
      for y in range(h - 1 - self.default_frame[1], h):
        mean = img[y:y + 1, 0:w].mean()
        if mean <= limit:
          img = img[0:y, 0:w]
          h, w = img.shape
          break
      if mean <= limit:
        break
    for limit in limits:
      for x in range(self.default_frame[2], 0, -1):
        # mean = img[0:h, x:x + 1].mean()
        mean = max([img[_y:_y + h // 4, x:x + 1].mean() for _y in range(0, h, h // 4)])
        if mean <= limit:
          img = img[0: h, x: w]
          h, w = img.shape
          break
      if mean <= limit:
        break
    for limit in limits:
      for x in range(w - 1 - self.default_frame[3], w):
        mean = max([img[_y:_y + h // 4, x:x + 1].mean() for _y in range(0, h, h // 4)])
        if mean <= limit:
          img = img[0:h, 0:x]
          h, w = img.shape
          break
      if mean <= limit:
        break
    if DEBUG:
      nimg = img.copy()
      cv2.imwrite(file_nimg, nimg)
      cv2.imwrite(file_bimg, img)
    else:
      cv2.imwrite(file, img)
    return 1

  def fill_black_holes(self, thresh, fill_hole=6, iteration=16):
    invert_fill_hole = True
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=iteration)
    if invert_fill_hole:
      thresh = 255 - thresh
    return thresh
  def fill_white_holes(self, thresh, fill_hole=4, iteration=16):
    invert_fill_hole = False
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, fill_hole), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=16)
    if invert_fill_hole:
      thresh = 255 - thresh
    return thresh
  def fill_countours_white(self, img, lower_limit=10, upper_limit=400):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # fill contours with white
    for i, contour in enumerate(contours):
      x, y, w, h = cv2.boundingRect(contour)
      if w > lower_limit and h > lower_limit and w < upper_limit and h < upper_limit:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img
  def center_block(self):
    self.default_frame = (1000,1000,1000,1000)
    file = self.original_image
    file_bimg = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    file_nimg = '.'.join(file.split('.')[:-1]) + '_nimg.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    oh, ow = img.shape
    isfirst = False
    if self.model_2 is not None:
      _img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
      if ow > oh:
        crop = _img[0:1200, ow // 2:, :]
      else:
        crop = _img[0:1200, :, :]
      crop = cv2.resize(crop, (224, 224), Image.Resampling.LANCZOS)
      cv2.imwrite(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
      crop = Image.open(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'))
      isfirst = self.newspaper.is_first_page(crop, self.model_2)
      # if isfirst:
      #   dest = '/home/sunvod/sormani_CNN/firstpage'
      #   _, _, files = next(os.walk(dest))
      #   file_count = len(files) + 1
      #   os.rename(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), os.path.join(dest, 'img_no__' + str(file_count) + '.jpeg'))
      # else:
      #   dest = '/home/sunvod/sormani_CNN/nofirstpage'
      #   _, _, files = next(os.walk(dest))
      #   file_count = len(files) + 1
      #   os.rename(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), os.path.join(dest, 'img__' + str(file_count) + '.jpeg'))
    ret, thresh = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
    _img = img.copy()
    limit = 250
    ofset = 800
    if isfirst:
      dim = 100
    else:
      dim = 30
    for y1 in range(self.default_frame[0], 0, -10):
      if ow > oh:
        mean = min(thresh[y1:y1 + dim, ofset:ow // 2].mean(), thresh[y1:y1 + dim, ow // 2: ow - ofset].mean()) + 2
      else:
        mean = thresh[y1:y1 + dim, ofset:ow-ofset].mean()
      if mean >= limit:
        break
    for y2 in range(oh - 1 - self.default_frame[1], oh, 10):
      mean = thresh[y2:y2 + dim, ofset:ow-ofset].mean()
      if mean >= limit:
        break
    for x1 in range(self.default_frame[2], 0, -10):
      mean = thresh[ofset:oh-ofset, x1:x1 + dim].mean()
      if mean >= limit:
        break
    for x2 in range(ow - 1 - self.default_frame[3], ow, 10):
      mean = thresh[ofset:oh-ofset, x2:x2 + dim].mean()
      if mean >= limit:
        break
    y1 = y1 - 60 if y1 - 60 > 0 else 0
    y2 = y2 + 30 if y2 + 30 < oh else oh
    x1 = x1 - 30 if x1 - 30 > 0 else 0
    x2 = x2 + 30 if x2 + 30 < ow else ow
    lx, ly, x_ofset, y_ofset = self.newspaper.get_limits()
    if ow < oh:
      lx = lx // 2
    upper_edge = 170
    lower_edge = 200
    if y2 - y1 < ly - y_ofset:
      y1 = 0
      y2 = oh
      upper_edge = 0
      lower_edge = 0
    if not self.only_x  or oh > ly:
      img = img[y1:y2, :]
      img = cv2.copyMakeBorder(img, upper_edge, lower_edge, 0, 0, cv2.BORDER_CONSTANT, value=self.color)
    if x2 - x1 > lx - x_ofset:
      img = img[ :, x1:x2]
      img = cv2.copyMakeBorder(img, 0, 0, 200, 200, cv2.BORDER_CONSTANT, value=self.color)
    # img = img[y1:y2, x1:x2]
    # img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=self.color)
    if DEBUG:
      nimg = img.copy()
      cv2.imwrite(file_nimg, nimg)
      cv2.imwrite(file_bimg, img)
    else:
      cv2.imwrite(file, img)
    return 1
  def rotate_final_frames(self):
    def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    def _ordering_contours(c):
      x, y, w, h = cv2.boundingRect(c)
      return (x, w)
    count = 0
    file = self.original_image
    file_bing = '.'.join(file.split('.')[:-1]) + '_bing.' + file.split('.')[-1]
    file_ning = '.'.join(file.split('.')[:-1]) + '_ning.' + file.split('.')[-1]
    file_thresh = '.'.join(file.split('.')[:-1]) + '_thresh.' + file.split('.')[-1]
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    if self.angle is not None:
      img = rotate_image(img, self.angle)
      cv2.imwrite(file, img)
      return 1
    ret, thresh = cv2.threshold(img, 32, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if w < 5 or h < 5:
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), -1)    # Questo elimina i punti < 3
    fill_hole = 16
    invert_fill_hole = False
    if invert_fill_hole:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
      thresh, binaryImage = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((fill_hole, 8), np.uint8)
    thresh = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=16)
    if invert_fill_hole:
      thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bimg = img.copy()
    bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
    rect = None
    _contours = []
    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      if (x > 0 or y > 0) and w > self.limit and h > self.limit:
        _contours.append(contour)
    # points = [(x, y), (x + w - 1, y), (x + w - 1, y + h -1 ), (x, y + h - 1)]
    # contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    _contours.sort(key=_ordering_contours)
    contours = _contours
    if len(_contours):
      contour = max(contours, key=cv2.contourArea)
      rect = cv2.minAreaRect(contour)
      if DEBUG:
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(bimg, [box], -1, (0, 0, 255), 3)
    if rect is not None:
      angle = rect[2]
      if angle < 45:
        angle = 90 + angle
      if angle > 85 and (angle < 89.9 or angle > 90.1):
        angle = angle - 90
        if angle < 5.0:
          img = rotate_image(img, angle)
          count += 1
      if DEBUG:
        cv2.imwrite(file_ning, img)
        cv2.imwrite(file_bing, bimg)
        cv2.imwrite(file_thresh, thresh)
      else:
        cv2.imwrite(file, img)
    return count


