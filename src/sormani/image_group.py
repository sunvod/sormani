from __future__ import annotations

import pathlib
import datetime
import re

from os import listdir

from src.sormani.page import Page
from src.sormani.page_pool import Page_pool
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import warnings
warnings.filterwarnings("ignore")

class Images_group():

  def __init__(self,  newspaper_base, newspaper_name, filedir, files):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.files = files
    year = ''.join(filter(str.isdigit, filedir.split('/')[-3]))
    month = ''.join(filter(str.isdigit, filedir.split('/')[-2]))
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
    self.newspaper = Newspaper.create(self.newspaper_name, os.path.join(filedir, files[0]), newspaper_base, self.date)
  def get_page_pool(self, newspaper_name, root, ext, image_path, path_exist, force):
    page_pool = Page_pool(newspaper_name, self.filedir.split('/')[-1], self.date, force)
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
          if not(Path(file).stem + '.pdf' in listdir(filedir)):
            exists = False
            break
      if exists:
        return page_pool
    for file in self.files:
      if pathlib.Path(file).suffix == '.' + ext:
        page = Page(Path(file).stem, self.date, self.newspaper, os.path.join(self.filedir, file), dir_path, dir_path, txt_path)
        page_pool.append(page)
    return page_pool

