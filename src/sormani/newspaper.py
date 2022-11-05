
from __future__ import annotations

import os
import datetime
from PIL import Image
from pathlib import Path

from src.sormani.const import MONTHS, exec_ocrmypdf


class Newspaper():
  @staticmethod
  def create(file_name, name, date):
    if name == 'La Stampa':
      newspaper = La_stampa(file_name, date)
    elif name == 'Il Manifesto':
      newspaper = Il_manifesto(file_name, date)
    if name == 'Avvenire':
      newspaper = Avvenire(file_name, date)
    return newspaper
  def __init__(self, name, file_name, date):
    self.name = name
    self.file_name = file_name
    self.date = date
    self.year, self.number = self.get_head()
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
  def crop(self, left, top, right, bottom):
    image = Image.open(self.file_name)
    image = image.crop((left, top, right, bottom))
    image.save('temp.tif')
    exec_ocrmypdf('temp.tif', oversample=800)
    f = open("temp.txt", "r")
    x = f.read()
    print(x)
    os.remove('temp.tif')
    os.remove('temp.pdf')
    os.remove('temp.txt')
    return x[:-1]

class La_stampa(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'La Stampa', file_name, date)
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
    # text = super().crop(left = 920, top = 1550, right = 1260, bottom = 1650)
    # year = ''.join(filter(str.isdigit, text[4:9]))
    # number = ''.join(filter(str.isdigit, text[-4:]))
    year = str(150 + self.date.year - 2016)
    number = str((self.date - datetime.date(self.date.year, 1, 1)).days)
    return year, number

class Il_manifesto(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'Il Manifesto', file_name, date)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    self.n_page = n_page + 1
class Avvenire(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'Avvenire', file_name, date)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    self.n_page = n_page + 1
