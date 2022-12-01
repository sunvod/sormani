
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

class Newspaper_parameters():
  def __init__(self, scale, max_perimeter, min_w, max_w, min_h, max_h, ts ):
    self.scale = scale
    self.max_perimeter = max_perimeter
    self.box = (min_w, max_w, min_h, max_h)
    self.ts = ts

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
    else:
      error = "Error: \'" + name + "\' is not defined in this application."
      raise ValueError(error)
    return newspaper
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

class La_stampa(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 150
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'La Stampa', file_path, date, year, number, init_page = 3)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 500)
    return whole
  def get_page_location(self):
    left = [0, 100, 700, 500]
    right = [4100, 100, 4800, 500]
    return left, right
  def get_page(self):
    left, right = self.get_page_location()
    text1, image1, dimension1 = super().crop(left=right[0], top=right[1], right=right[2], bottom=right[3])
    n1 = ''.join(filter(str.isdigit, text1))
    text2, image2, dimension2 = super().crop(left=left[0], top=left[1], right=left[2], bottom=left[3])
    n2 = ''.join(filter(str.isdigit, text2))
    if n1.isdigit() and n2.isdigit():
      if dimension1 > dimension2:
        return n1, [image1, image2]
      else:
        return n2, [image1, image2]
    if n1.isdigit():
      return n1, [image1, image2]
    elif n2.isdigit():
      return n2, [image1, image2]
    else:
      return '??', [image1, image2]
  # def crop_png(self, image):
  #   # whole = (0, 100, 5000, 500)
  #   w, h = image.size
  #   if w < 2000:
  #     return image
  #   image = image.crop(self.get_whole_page_location(image))
  #   image1 = image.crop((200, 100, 500, h - 100))
  #   image2 = image.crop((w - 500, 100, w - 200, h - 100))
  #   image = Image.new('RGB', (600, h))
  #   image.paste(image1, (0, 0))
  #   image.paste(image2, (300, 0))
  #   return image
  def crop_png(self, image):
    image = image.crop(self.get_whole_page_location(image))
    return image
  def get_parameters(self):
    return 200, 200, 20, 120, 60, 170, 64

class Il_Giornale(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 150
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Giornale', file_path, date, year, number, init_page = 3)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_whole_page_location(self, image):
    whole = (0, 100, 5000, 500)
    return whole
  def get_page_location(self):
    left = [0, 100, 700, 500]
    right = [4100, 100, 4850, 500]
    return left, right
  def get_page(self):
    return None, None

class Il_manifesto(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 46
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Manifesto', file_path, date, year, number, init_page = 3)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left = [0, 100, 700, 500]
    right =  [4100, 100, 4850, 500]
    return left, right
  def get_whole_page_location(self, image):
    whole = [0, 150, 4850, 450]
    return whole
  def get_page(self):
    left, right = self.get_page_location()
    text1, image1, dimension1 = super().crop(left=right[0], top=right[1], right=right[2], bottom=right[3])
    n1 = ''.join(filter(str.isdigit, text1))
    text2, image2, dimension2 = super().crop(left=left[0], top=left[1], right=left[2], bottom=left[3])
    n2 = ''.join(filter(str.isdigit, text2))
    if n1.isdigit() and n2.isdigit():
      if dimension1 > dimension2:
        return n1, [image1, image2]
      else:
        return n2, [image1, image2]
    if n1.isdigit():
      return n1, [image1, image2]
    elif n2.isdigit():
      return n2, [image1, image2]
    else:
      return '??', [image1, image2]
  # def crop_png(self, image):
  #   image = image.crop(self.get_whole_page_location(image))
  #   return image
  def crop_png(self, image):
    w, h = image.size
    if w < 2000:
      return
    box = self.get_whole_page_location(image)
    file_name = Path(self.file_path).stem
    n = (Path(file_name).stem).split('_')[-1][-1:]
    if int(n) % 2 == 0:
      image = image.crop((box[0] + 100, box[1], box[0] + 700, box[3]))
    else:
      image = image.crop((box[2] - 700, box[1], box[2] - 100, box[3]))
    return image
  def get_parameters(self):
    # return 100, 20, 60, 55, 70
    return 200, 100, 5, 100, 20, 100, 64

class Milano_Finanza(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 27
    self.year_change = [1, 9]
    Newspaper.__init__(self, newspaper_base, 'Milano Finanza', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    if n_page >= 2:
      if (n_page - 2) % 4 == 0:
        n_page += 1
      elif (n_page - 3) % 4 == 0:
        n_page += 1
      elif (n_page - 4) % 4 == 0:
        n_page -= 2
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    #print(self.n_page, '  ', end='')
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None
    # left, right = self.get_page_location()
    # text1, image1, dimension1 = super().crop(left=right[0], top=right[1], right=right[2], bottom=right[3])
    # n1 = ''.join(filter(str.isdigit, text1))
    # text2, image2, dimension2 = super().crop(left=left[0], top=left[1], right=left[2], bottom=left[3])
    # n2 = ''.join(filter(str.isdigit, text2))
    # if n1.isdigit() and n2.isdigit():
    #   if dimension1 > dimension2:
    #     return n1, [image1, image2]
    #   else:
    #     return n2, [image1, image2]
    # if n1.isdigit():
    #   return n1, [image1, image2]
    # elif n2.isdigit():
    #   return n2, [image1, image2]
    # else:
    #   return '??', [image1, image2]
class Il_Fatto_Quotidiano(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 8
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Il Fatto Quotidiano', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None
class Italia_Oggi(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 25
    self.year_change = [8, 8]
    Newspaper.__init__(self, newspaper_base, 'Italia Oggi', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages=None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    if n_page >= 2:
      if (n_page - 2) % 4 == 0:
        n_page += 1
      elif (n_page - 3) % 4 == 0:
        n_page += 1
      elif (n_page - 4) % 4 == 0:
        n_page -= 2
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    # print(self.n_page, '  ', end='')
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None
class Libero(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 51
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Libero', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None
class Alias(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 19
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Alias', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None

class Alias_Domenica(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 6
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Alias Domenica', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    whole = [0, 100, 4850, 400]
    return whole
  def get_page(self):
    return None, None

class Avvenire(Newspaper):
  def __init__(self, newspaper_base, file_path, date, year, number):
    self.init_year = 49
    self.year_change = None
    Newspaper.__init__(self, newspaper_base, 'Avvenire', file_path, date, year, number, init_page = 5)
  def set_n_page(self, n_page, date, pages = None):
    if super().check_n_page(date):
      self.n_page = n_page + 1
      return
    if n_page >= self.n_pages:
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
    pass
  def get_page_location(self):
    left =  [4100, 100, 4850, 400]
    right = [0, 100, 700, 400]
    return [left, right]
  def get_whole_page_location(self, image):
    w, h = image.size
    whole = (0, 100, w, 800)
    return whole
  def get_page(self):
    return None, None
  def crop_png(self, image):
    image = image.crop(self.get_whole_page_location(image))
    return image
  def get_parameters(self):
    return Newspaper_parameters(200, 200, 100, 400, 100, 400, 16)
    # return 200, 200, 100, 400, 200, 400, 16
