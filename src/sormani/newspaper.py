
from __future__ import annotations

import os
import datetime
from PIL import Image
from pathlib import Path

from src.sormani.system import MONTHS, exec_ocrmypdf


class Newspaper():
  @staticmethod
  def create(newspaper_base, file_name, name, date, year = None, number = None):
    if name == 'La Stampa':
      newspaper = La_stampa(newspaper_base, file_name, date, year, number)
    elif name == 'Il Manifesto':
      newspaper = Il_manifesto(newspaper_base, file_name, date, year, number)
    if name == 'Avvenire':
      newspaper = Avvenire(newspaper_base, file_name, date, year, number)
    return newspaper
  def __init__(self, newspaper_base, name, file_name, date, year, number):
    self.newspaper_base = newspaper_base
    self.name = name
    self.file_name = file_name
    self.date = date
    self.contrast = 50
    if year is not None:
      self.year = year
      self.number = number
    else:
      self.year, self.number = self.get_head()
    self.page = None
  def set_n_page(self, n_page, date):
    file_name = Path(self.file_name).stem
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
  def get_number(self, date, week_day = None):
    start = datetime.date(date.year, 1, 1)
    num_weeks, remainder = divmod((date - start).days, 7)
    num_days = (date - start).days + 1
    if week_day is not None and (week_day - start.weekday()) % 7 < remainder:
      num_weeks += 1
    num_days -= num_weeks
    return num_days
  def change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def crop(self, left, top, right, bottom, resize = None, contrast = None):
    image = Image.open(self.file_name)
    image = image.crop((left, top, right, bottom))
    if resize is not None:
      image = image.resize((int(image.size[0] * resize), int(image.size[1] * resize)), Image.Resampling.LANCZOS)
    if contrast is not None:
      image = self.change_contrast(image, contrast)
    image.save('temp.tif')
    exec_ocrmypdf('temp.tif', oversample=800)
    f = open("temp.txt", "r")
    x = f.read()
    x = x.replace("\n", "").strip()
    print(x)
    image = Image.open('temp.tif')
    os.remove('temp.tif')
    os.remove('temp.pdf')
    os.remove('temp.txt')
    return x, image
  def get_number(self):
    dir = self.file_name
    folder_count = 0
    for month in range(self.date.month):
      input_path = os.path.join(self.newspaper_base, str(self.date.year), str(month + 1))
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

class La_stampa(Newspaper):
  def __init__(self, newspaper_base, file_name, date, year, number):
    Newspaper.__init__(self, newspaper_base, 'La Stampa', file_name, date, year, number)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
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
  def get_head(self):
    #text = super().crop(left = 940, top = 1500, right = 1300, bottom = 1700)
    # year = ''.join(filter(str.isdigit, text[4:9]))
    #number = ''.join(filter(str.isdigit, text[12:14]))
    number = self.get_number()
    year = str(150 + self.date.year - 2016)
    return year, number
  def get_page(self):
    text1, image1 = super().crop(left=4000, top=100, right=5000, bottom=500)
    n1 = ''.join(filter(str.isdigit, text1))
    text2, image2 = super().crop(left=0, top=100, right=1000, bottom=500)
    n2 = ''.join(filter(str.isdigit, text2))
    if n1.isdigit():
      return n1, image1
    elif n2.isdigit():
      return n2, image2
    else:
      return '??', [image1, image2]

class Il_manifesto(Newspaper):
  def __init__(self, newspaper_base, file_name, date, year, number):
    Newspaper.__init__(self, newspaper_base, 'Il Manifesto', file_name, date, year, number)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
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
  def get_head(self):
    text = super().crop(left = 1250, top = 1450, right = 1500, bottom = 1550)
    number = ''.join(filter(str.isdigit, text))
    year = str(46 + self.date.year - 2016)
    number = self.get_number(self.date, 0)
    return year, number
class Avvenire(Newspaper):
  def __init__(self, newspaper_base, file_name, date, year, number):
    Newspaper.__init__(self, newspaper_base, 'Avvenire', file_name, date, year, number)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    self.n_page = n_page + 1
