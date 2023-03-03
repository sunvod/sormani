from __future__ import annotations

import pathlib
import datetime
import re
import cv2

from os import listdir

from src.sormani.page import Page
from src.sormani.page_pool import Page_pool
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import warnings
warnings.filterwarnings("ignore")

class Images_group():

  def __init__(self,  newspaper_base, newspaper_name, filedir, files, new_root):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.files = files
    self.new_root = new_root
    year = ''.join(filter(str.isdigit, filedir.split('/')[-3]))
    i = -2
    if not len(year):
      year = ''.join(filter(str.isdigit, filedir.split('/')[-2]))
      month = ''.join(filter(str.isdigit, filedir.split('/')[-1]))
      day = '01'
      if not len(year):
        error = 'Non esiste una directory per l\'anno.'
        raise OSError(error)
    else:
      month = ''.join(filter(str.isdigit, filedir.split('/')[i]))
      day_folder = filedir.split('/')[-1]
      pos = re.search(r'[^0-9]', day_folder + 'a').start()
      day = day_folder[ : pos]
    if year.isdigit() and month.isdigit() and day.isdigit():
      try:
        self.date = datetime.date(int(year), int(month), int(day))
      except:
        error = 'La directory \'' + year + '/' + month + '/' + day + '\' non rappresenta un giorno valido.'
        raise OSError(error)
    else:
      error = 'La directory \'' + year + '/' + month + '/' + day + '\' non rappresenta un giorno valido.'
      raise OSError(error)
    self.newspaper = Newspaper.create(self.newspaper_name, os.path.join(filedir, files[0]), newspaper_base, self.date, month = month)
  def get_page_pool(self, newspaper_name, new_root, ext, image_path, path_exist, force, thresholding, model, use_ai):
    page_pool = Page_pool(newspaper_name, self.filedir, self.filedir.split('/')[-1], new_root, self.date, force, thresholding, model)
    page_pool.isins = not self.filedir.split('/')[-1].isdigit()
    dir_in_filedir = self.filedir.split('/')
    txt_in_filedir = list(map(lambda x: x.replace(image_path, 'txt'), dir_in_filedir))
    dir_in_filedir = list(map(lambda x: x.replace(image_path, JPG_PDF_PATH), dir_in_filedir))
    dir_path = '/'.join(dir_in_filedir)
    txt_path = '/'.join(txt_in_filedir)
    filedir = os.path.join(dir_path, path_exist)
    if os.path.isdir(filedir) and not force:
      exists = True
      for file in self.files:
        if pathlib.Path(file).suffix == '.' + ext:
          if not (Path(file).stem + '.pdf' in listdir(filedir)):
            exists = False
            break
      if exists:
        return page_pool
    for file in self.files:
      if pathlib.Path(file).suffix == '.' + ext:
        page = Page(Path(file).stem, self.date, self.newspaper, page_pool.isins, os.path.join(self.filedir, file), dir_path, dir_path, txt_path, model)
        if use_ai and model is not None and not page.isAlreadySeen():
          if page.file_name.split('_')[-1] == '0' or dir_path.split('/')[-1].split()[-1][0:2] == 'OT':
            image = Image.open(page.original_image)
            w, h = image.size
            if h > w:
              page.prediction = page.get_prediction()
        page_pool.append(page)
    return page_pool

