from __future__ import annotations

import pathlib
import datetime
import re
import cv2

from os import listdir

from PIL.Image import DecompressionBombError

from src.sormani.page import Page
from src.sormani.page_pool import Page_pool
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import warnings
warnings.filterwarnings("ignore")

class Images_group():

  def __init__(self,  newspaper_base, newspaper_name, filedir, files, is_bobina):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.files = files
    self.is_bobina = is_bobina
    # self.new_root = new_root
    year = ''.join(filter(str.isdigit, filedir.split('/')[-3]))
    i = -2
    if not is_bobina:
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
    else:
      self.date = None
      month = None
    self.newspaper = Newspaper.create(self.newspaper_name, os.path.join(filedir, files[0]), newspaper_base, self.date, month = month, is_bobina=is_bobina)
  def is_image(self, filedir, file):
    try:
      img = cv2.imread(os.path.join(filedir, file), cv2.IMREAD_UNCHANGED)
    except DecompressionBombError:
      pass
    except:
      img = None
    if img is None:
      with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
        sormani_log.write('No valid Image: ' + os.path.join(filedir, file) + '\n')
      print(f'Not a valid image: {os.path.join(filedir, file)}')
      return False
    if img.ndim > 2 and img.shape[2] == 4:
      img = img[:,:,:3]
      cv2.imwrite(os.path.join(filedir, file), img)
    return True
  def get_page_pool(self, newspaper_name, new_root, ext, image_path, path_exist, force, thresholding, ais, checkimages, is_bobina=False):
    page_pool = Page_pool(newspaper_name, self.filedir, self.filedir.split('/')[-1], new_root, self.date, force, thresholding, ais)
    page_pool.isins = not self.filedir.split('/')[-1].isdigit() if not is_bobina else False
    dir_in_filedir = self.filedir.split('/')
    txt_in_filedir = list(map(lambda x: x.replace(image_path, 'txt'), dir_in_filedir))
    dir_in_filedir = list(map(lambda x: x.replace(image_path, JPG_PDF_PATH), dir_in_filedir))
    # if is_bobina:
    #   _dir_in_filedir = dir_in_filedir[-1]
    #   dir_in_filedir = dir_in_filedir[:-3]
    #   dir_in_filedir.append(_dir_in_filedir)
    #   _txt_in_filedir = txt_in_filedir[-1]
    #   txt_in_filedir = txt_in_filedir[:-3]
    #   txt_in_filedir.append(_txt_in_filedir)
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
        if checkimages and not self.is_image(self.filedir, file):
          continue
        page = Page(Path(file).stem, self.date, self.newspaper, page_pool.isins, os.path.join(self.filedir, file), dir_path, dir_path, txt_path, ais, is_bobina)
        ai = ais.get_model(PAGE)
        model = ai.model if ai is not None else None
        use_ai = ai.use if ai is not None else False
        if use_ai and model is not None and not page.isAlreadySeen():
          if page.file_name.split('_')[-1] == '0' or dir_path.split('/')[-1].split()[-1][0:2] == 'OT':
            image = Image.open(page.original_image)
            w, h = image.size
            if h > w:
              page.prediction = page.get_prediction()
        ai = ais.get_model(ISFIRSTPAGE)
        model = ai.model if ai is not None else None
        use_ai = ai.use if ai is not None else False
        if use_ai and model is not None and page.newspaper.is_first_page() is None:
          img = cv2.imread(page.original_image)
          isfirst, crop = page.newspaper.is_first_page(img, model, get_crop=True)
          file_name = file.split('.')[0]
          if ai.save:
            if isfirst:
              dest = '/home/sunvod/sormani_CNN/firstpage'
              Path(dest).mkdir(parents=True, exist_ok=True)
              os.rename(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), os.path.join(dest, str(self.date)  + '_' + file_name+ '.jpeg'))
            else:
              dest = '/home/sunvod/sormani_CNN/nofirstpage'
              Path(dest).mkdir(parents=True, exist_ok=True)
              os.rename(os.path.join(STORAGE_BASE, 'img_jpg' + '.jpg'), os.path.join(dest, str(self.date)  + '_' + file_name+ '.jpeg'))
        page_pool.append(page)
    ais.garbage_model(PAGE)
    ais.garbage_model(ISFIRSTPAGE)
    page_pool.sort(key=lambda x: x.original_image)
    return page_pool

