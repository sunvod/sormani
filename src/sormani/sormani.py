import multiprocessing
import time
import datetime
import os
import imghdr
import portalocker
import pathlib

import tkinter as tk
from PIL import Image, ImageTk
from src.sormani.page import Images_group
from multiprocessing import Pool
import numpy as np

from src.sormani.system import IMAGE_PATH, IMAGE_ROOT, N_PROCESSES, STORAGE_DL, JPG_PDF_PATH

global_count = multiprocessing.Value('I', 0)

class Conversion:
  def __init__(self, image_path, dpi, quality, resolution):
    self.image_path = image_path
    self.dpi = dpi
    self.quality = quality
    self.resolution = resolution

class Sormani():
  def __init__(self,
               newspaper_name,
               root = IMAGE_ROOT,
               year = None,
               months = None,
               days = None,
               ext = 'tif',
               image_path = IMAGE_PATH,
               path_exclude = [],
               path_exist ='pdf',
               force = False,
               contrast = True):
    if not isinstance(months, list):
      months = [months]
    if not isinstance(days, list):
      days = [days]
    elements = []
    for month in months:
      for day in days:
        elements.append(self._init(newspaper_name, root, year, month, day, ext, image_path, path_exclude, path_exist, force, contrast))
    self.elements = []
    for element in elements:
      for e in element:
        self.elements.append(e)
    pass
  def _init(self, newspaper_name, root, year, month, day, ext, image_path, path_exclude, path_exist, force, contrast):
    self.newspaper_name = newspaper_name
    self.root = root
    self.i = 0
    self.elements = []
    root = os.path.join(root, image_path, newspaper_name)
    self.add_zero_to_dir(root)
    if not os.path.exists(root):
      print(f'{newspaper_name} non esiste in memoria.')
      return self.elements
    if year is not None:
      root = os.path.join(root, self.add_zero(year))
      if not os.path.exists(root):
        print(f'{newspaper_name} per l\'anno {year} non esiste in memoria.')
        return self.elements
      if month is not None:
        root = os.path.join(root, self.add_zero(month))
        if not os.path.exists(root):
          print(f'{newspaper_name} per l\'anno {year} e per il mese {month} non esiste in memoria.')
          return self.elements
        if day is not None:
          root = os.path.join(root, self.add_zero(day))
          if not os.path.exists(root):
            print(f'{newspaper_name} per l\'anno {year}, per il mese {month} e per il giorno {day} non esiste in memoria.')
            return self.elements
    self.ext = ext
    self.image_path = image_path
    self.path_exclude = path_exclude
    self.path_exist = path_exist
    self.force = force
    self.converts = None
    self.elements = self.get_elements(root)
    self.new_root = root
    if contrast:
      self.change_all_contrasts()
    self.divide_all_image()
    self.elements = self.get_elements(root)
    self.set_all_images_names()
    self.elements = self.get_elements(root)
    return self.elements
    pass
  def count_pages(self):
    count = 0
    for element in self.elements:
      count += len(element.files)
    return count
  def count_number(self):
    if len(self.elements) == 0:
      return 'ND'
    page_pool = self.elements[self.i].get_page_pool(self.newspaper_name, self.root, self.ext, self.image_path, self.path_exist, self.force)
    if not page_pool.isAlreadySeen():
      pages = page_pool.extract_pages(range=(3, 4))
      pages = int(pages[0]) + 1 if isinstance(pages, list) and len(pages) > 0 and pages[0].isdigit() and int(
        pages[0]) + 1 < len(page_pool) else len(page_pool)
    else:
      pages = len(page_pool)
    page_pool.set_pages(pages)
    self.i += 1
    return page_pool[0].newspaper.number
  def add_zero(self, n):
    if isinstance(n, int):
      n = str(n)
    if n.isdigit() and len(n) == 1:
      n = '0' + n
    return n
  def add_zero_to_dir(self, root):
    for filedir, dirs, files in os.walk(root):
      for dir in dirs:
        if dir.isdigit() and len(dir) == 1:
          os.rename(os.path.join(filedir, dir), os.path.join(filedir, '0' + dir))
  def get_elements(self, root):
    elements = []
    for filedir, dirs, files in os.walk(root):
      n_pages = len(files)
      if filedir in self.path_exclude or n_pages == 0:
        continue
      files.sort(key = self._get_elements)
      if self.check_if_image(filedir, files):
        elements.append(Images_group(os.path.join(self.root, self.image_path, self.newspaper_name), self.newspaper_name, filedir, files, get_head = True))
    if len(elements) > 1:
      elements.sort(key=self._elements_sort)
    return elements
  def check_if_image(self, filedir, files):
    for file_name in files:
      try:
        Image.open(os.path.join(filedir, file_name))
      except:
      #if not imghdr.what(os.path.join(filedir, file_name)):
        with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
          sormani_log.write('No valid Image: ' + os.path.join(filedir, file_name) + '\n')
        print(f'Not a valid image: {os.path.join(filedir, file_name)}')
        return False
    return True
  def _get_elements(self, n):
    # n = e[:5]
    n = ''.join(c for c in n if c.isdigit())
    if not len(n):
      n = '0'
    r = ['0' for x in range(25 - len(n))]
    r = ''.join(r) + n
    return r
  def __len__(self):
    return len(self.elements)
  def __iter__(self):
    return self
  def __next__(self):
    if self.i < len(self.elements):
      page_pool = self.elements[self.i].get_page_pool(self.newspaper_name, self.root, self.ext, self.image_path, self.path_exist, self.force)
      if not page_pool.isAlreadySeen():
        if len(page_pool) > 0:
          init_page = page_pool[0].newspaper.init_page
          pages = page_pool.extract_pages(range=(init_page, init_page + 1))
          pages = int(pages[0]) + 1 \
            if len(pages) > 0 and isinstance(pages, list) and pages[0] is not None and len(pages) > 0 and pages[0].isdigit() and int(pages[0]) + 1 < len(page_pool) \
            else len(page_pool)
        else:
          pages = len(page_pool)
      else:
        pages = len(page_pool)
      page_pool.set_pages(pages)
      self.i += 1
      return page_pool
    else:
      self.i = 0
      raise StopIteration
  def _elements_sort(self, images_group):
    return images_group.date
  def set_force(self, force):
    self.force = force
  def create_all_images(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)]):
    if not len(self.elements):
      return
    for page_pool in self:
      if not len(page_pool):
        continue
      page_pool.create_pdf()
      page_pool.convert_images(converts)
  def set_all_images_names(self):
    if not len(self.elements):
      return
    for page_pool in self:
      if not len(page_pool):
        continue
      page_pool.set_image_file_name()
  def change_all_contrasts(self, contrast = None, force = False):
    if not len(self.elements):
      return
    start_time = time.time()
    print(f'Start changing the contrast of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = force
    global global_count
    global_count.value = 0
    self.contrast = contrast
    self.force = force
    with Pool(processes=N_PROCESSES) as mp_pool:
      mp_pool.map(self.change_contrast, self)
    if global_count.value:
      print()
      print(f'It has changed the contrast of {global_count.value} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no images to change the contrast for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def change_contrast(self, page_pool):
    global global_count
    count = 0
    #print(f'Changing contrast to \'{page_pool.newspaper_name}\' of {page_pool.date.strftime("%d/%m/%y")}')
    for page in page_pool:
      page.contrast = self.contrast
      page.force = self.force
      i = page.change_contrast()
      count += i
      with global_count.get_lock():
        global_count.value += i
    if count:
      print(',', end='')

  def divide_all_image(self):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(
      f'Starting division of \'{self.newspaper_name}\' ({self.new_root}) in date {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    with Pool(processes=14) as mp_pool:
      mp_pool.map(self.divide_image, self.elements)
    if global_count.value:
      print()
      print(
        f'Division of {global_count.value} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'No division is needed for \'{self.newspaper_name}\'.')
    return

  def divide_image(self, image_group):
    i = 0
    for file_name in image_group.files:
      file_path = os.path.join(image_group.filedir, file_name)
      im = Image.open(file_path)
      width, height = im.size
      if width < height:
        continue
      left = 0
      top = 0
      right = width // 2
      bottom = height
      im1 = im.crop((left, top, right, bottom))
      im1.save(file_path[: len(file_path) - 4] + '-2.tif')
      left = width // 2 + 1
      top = 0
      right = width
      bottom = height
      im2 = im.crop((left, top, right, bottom))
      im2.save(file_path[: len(file_path) - 4] + '-1.tif')
      os.remove(file_path)
      i += 1
    if i:
      print(',', end='')
      with global_count.get_lock():
        global_count.value += i
  def create_jpg(self):
      for page_pool in self:
        page_pool.convert_images(converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)])
  def extract_pages(self, range = None, mute = True, image_mute = True):
    if not len(self.elements):
      return
    if not isinstance(range, tuple):
      if isinstance(range, int):
        range = (0, range)
      else:
        print('Error: range is not a tuple nor a integer.')
        return
    start_time = time.time()
    if not mute:
      print(f'Starting extracting page number from \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    pages = []
    for page_pool in self:
      last_pages = page_pool.extract_pages(mute, image_mute, range)
      pages.append(last_pages)
      count += len(last_pages)
    if not mute:
      if count:
        print(f'Extracting page number from {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
      else:
        print(f'Warning: No extraction of page number are made for \'{self.newspaper_name}\'.')
    self.force = selfforce
    return pages
  def save_pages_images(self):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start extract pages images of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    image_path = os.path.join(STORAGE_DL, 'train')
    for page_pool in self:
      count += page_pool.save_pages_images(image_path)
    if count:
      print(f'It has extracted {count} page images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no pages images to extract for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def move_pdf_txt(self):
    for filedir, dirs, files in os.walk('/mnt/storage01/sormani/TIFF'):
      if len(files):
        for file in files:
          ext = pathlib.Path(file).suffix
          if ext == '.pdf' or ext == '.txt':
            dir_in_filedir = filedir.split('/')
            new_filedir = list(
              map(lambda x: x.replace('TIFF', 'txt' if ext == '.txt' else JPG_PDF_PATH), dir_in_filedir))
            new_path = '/'.join(new_filedir)
            if ext == '.pdf':
              new_path = os.path.join(new_path, 'pdf')
            new_file = os.path.join(new_path, file)
            old_file = os.path.join(filedir, file)
            os.rename(old_file, new_file)
            pass


