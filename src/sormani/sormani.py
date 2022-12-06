import multiprocessing
import time
import datetime
import os
import imghdr
import portalocker
import pathlib
import re

import tkinter as tk
from PIL import Image, ImageTk
from src.sormani.page import Images_group
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from src.sormani.system import IMAGE_PATH, IMAGE_ROOT, N_PROCESSES, STORAGE_DL, JPG_PDF_PATH, N_PROCESSES_SHORT

import warnings
warnings.filterwarnings("ignore")

global_count = multiprocessing.Value('I', 0)

class Conversion:
  def __init__(self, image_path, dpi, quality, resolution):
    self.image_path = image_path
    self.dpi = dpi
    self.quality = quality
    self.resolution = resolution

class Sormani():
  def __init__(self,
               newspaper_names,
               root = IMAGE_ROOT,
               year = None,
               months = None,
               days = None,
               ext = 'tif',
               image_path = IMAGE_PATH,
               path_exclude = [],
               path_exist ='pdf',
               force = False,
               rename_only = False,
               rename_file_names =  True):
    if not isinstance(newspaper_names, list):
      if newspaper_names is not None:
        name = newspaper_names
        newspaper_names = []
        newspaper_names.append(name)
      else:
        newspaper_names = ['La Stampa', 'Il Giornale', 'Il Manifesto', 'Avvenire', 'Milano Finanza', 'Il Fatto Quotidiano', 'Italia Oggi', 'Libero', 'Alias', 'Alias Domenica']
        rename_only = True
    if not isinstance(months, list):
      months = [months]
    if not isinstance(days, list):
      days = [days]
    elements = []
    for newspaper_name in newspaper_names:
      for month in months:
        for day in days:
          e = self._init(newspaper_name, root, year, month, day, ext, image_path, path_exclude, path_exist, force, rename_only, rename_file_names)
          if e is not None:
            elements.append(e)
    self.elements = []
    for element in elements:
      for e in element:
        self.elements.append(e)
    pass
  def _init(self, newspaper_name, root, year, month, day, ext, image_path, path_exclude, path_exist, force, rename_only, rename_file_names):
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
      if month is not None:
        root = os.path.join(root, self.add_zero(month))
        if day is not None:
          root = os.path.join(root, self.add_zero(day))
    self.rename_folder(root)
    if rename_only:
      return None
    self.ext = ext
    self.image_path = image_path
    self.path_exclude = path_exclude
    self.path_exist = path_exist
    self.force = force
    self.converts = None
    self.elements = self.get_elements(root)
    self.new_root = root
    self.divide_all_image()
    self.elements = self.get_elements(root)
    # if contrast:
    #   self.change_all_contrasts()
    # self.elements = self.get_elements(root)
    if rename_file_names:
      self.set_all_images_names()
    self.elements = self.get_elements(root)
    return self.elements
    pass
  def __len__(self):
    return len(self.elements)
  def __iter__(self):
    return self
  def __next__(self):
    if self.i < len(self.elements):
      page_pool = self.elements[self.i].get_page_pool(self.newspaper_name, self.root, self.ext, self.image_path, self.path_exist, self.force)
      if len(page_pool):
        init_page = int(page_pool[0].newspaper.number)
        pages = len(page_pool)  # page_pool.extract_pages(range=(init_page, init_page + 1))
        # pages = int(pages[0]) + 1 \
        #   if len(pages) > 0 and isinstance(pages, list) and pages[0] is not None and len(pages) > 0 and pages[0].isdigit() and int(pages[0]) + 1 < len(page_pool) \
        #   else len(page_pool)
        if page_pool.isAlreadySeen():
          page_pool.set_pages_already_seen(len(page_pool))
        else:
          page_pool.set_pages(pages)
      self.i += 1
      return page_pool
    else:
      self.i = 0
      raise StopIteration
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
      dirs.sort()
      for dir in dirs:
        p = re.search(r'[^0-9]', dir)
        if p is not None:
          p = p.start()
        else:
          p = len(dir)
        n = dir[:p]
        if n.isdigit() and len(n) == 1:
          try:
            os.rename(os.path.join(filedir, dir), os.path.join(filedir, '0' + dir))
            if len(files):
              dirs.remove(dir)
              dirs.append('0' + dir)
          except:
            pass
  def get_elements(self, root):
    elements = []
    for filedir, dirs, files in os.walk(root):
      n_pages = len(files)
      if filedir in self.path_exclude or n_pages == 0 or len(dirs) > 0:
        continue
      files.sort(key = self._get_elements)
      if self.check_if_image(filedir, files):
        elements.append(Images_group(os.path.join(self.root, self.image_path, self.newspaper_name), self.newspaper_name, filedir, files,))
    if len(elements) > 1:
      elements.sort(key=self._elements_sort)
    return elements
  def check_if_image(self, filedir, files):
    for file_name in files:
      try:
        Image.open(os.path.join(filedir, file_name))
      except:
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
    r = '000000000000000000000000000000' + n
    r = r[-30:]
    return r
  def _elements_sort(self, images_group):
    e = images_group.filedir.split('/')[-1]
    return e
  def set_force(self, force):
    self.force = force
  def create_all_images(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)], number = None, contrast = True):
    if not len(self.elements):
      return
    if contrast:
      self.change_all_contrasts()
    for page_pool in self:
      if not len(page_pool):
        continue
      page_pool.create_pdf(number)
      page_pool.convert_images(converts)
  def set_all_images_names(self):
    if not len(self.elements):
      return
    for page_pool in self:
      if not len(page_pool):
        continue
      else:
        if not page_pool.isAlreadySeen():
          init_page = int(page_pool[0].newspaper.number)
          pages = len(page_pool) # page_pool.extract_pages(range=(init_page, init_page + 1))
          # pages = int(pages[0]) + 1 \
          #   if len(pages) > 0 and isinstance(pages, list) and pages[0] is not None and len(pages) > 0 and pages[0].isdigit() and int(pages[0]) + 1 < len(page_pool) \
          #   else len(page_pool)
          page_pool.set_pages(pages)
          page_pool.set_image_file_name()
        else:
          page_pool.set_pages_already_seen(len(page_pool))
  def change_all_contrasts(self, contrast = None, force = False):
    if not len(self.elements):
      return
    start_time = time.time()
    print(f'Start changing the contrast of \'{self.newspaper_name}\' ({self.root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    global global_count
    global_count.value = 0
    self.contrast = contrast
    self.force = True
    with Pool(processes=N_PROCESSES_SHORT) as mp_pool:
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
    end = 0
    for page in page_pool:
      page.contrast = self.contrast
      page.force = self.force
      i = page.change_contrast()
      count += i
      with global_count.get_lock():
        global_count.value += i
        if count:
          print('.', end='')
          if global_count.value % 100 == 0:
            print()

  def divide_all_image(self):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(
      f'Starting division of \'{self.newspaper_name}\' ({self.new_root}) in date {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    with Pool(processes=N_PROCESSES_SHORT) as mp_pool:
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
    flag = False
    for file_name in image_group.files:
      file_path = os.path.join(image_group.filedir, file_name)
      im = Image.open(file_path)
      width, height = im.size
      if width > height:
        flag = True
        break
    if flag:
      for file_name in image_group.files:
        last = file_name.split('_')[-1]
        if last == '0' or last == '1' or last == '2':
          error = '\'' + file_name + '\' into folder \'' + image_group.filedir + '\' seems to be inconsistent because images into the folder are only partially divided.' \
                                                                                 '\nIt is necessary reload all the folder from backup data in order to have consistency again.'
          raise ValueError(error)
    for file_name in image_group.files:
      try:
        file_path = os.path.join(image_group.filedir, file_name)
        file_name_no_ext = Path(file_path).stem
        file_path_no_ext = os.path.join(image_group.filedir, file_name_no_ext)
        ext = Path(file_name).suffix
        im = Image.open(file_path)
        width, height = im.size
        if width < height:
          if flag:
            os.rename(file_path, file_path_no_ext + '_0' + ext)
          continue
        left = 0
        top = 0
        right = width // 2
        bottom = height
        im1 = im.crop((left, top, right, bottom))
        im1.save(file_path_no_ext + '_2' + ext)
        left = width // 2 + 1
        top = 0
        right = width
        bottom = height
        im2 = im.crop((left, top, right, bottom))
        im2.save(file_path_no_ext + '_1' + ext)
        os.remove(file_path)
        i += 1
      except:
        with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
          sormani_log.write('No valid Image: ' + file_path + '\n')
        print(f'Not a valid image: {file_path}')
    if i:
      print('.', end='')
      with global_count.get_lock():
        global_count.value += i
        if global_count.value % 100 == 0:
          print()

  def add_pdf_metadata(self, first_number = None):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(
      f'Start redefine Metadata of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = True
    self.first_number = first_number
    with Pool(processes=14) as mp_pool:
      mp_pool.map(self._add_pdf_metadata, self)
    if global_count.value:
      print(f'It has redefined Metadata of {global_count.value} ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no Metadata to be redefined for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def _add_pdf_metadata(self, page_pool):
    count = 0
    for page in page_pool:
      page.add_pdf_metadata(self.first_number)
      count += 1
    with global_count.get_lock():
      global_count.value += count
  def create_jpg(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)]):
      for page_pool in self:
        page_pool.convert_images(converts = converts)
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
  def get_pages_numbers(self, no_resize = False, filedir = None):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start extract pages numbers of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    images = []
    for page_pool in self:
      if not page_pool.isins:
        image = page_pool.get_pages_numbers(no_resize = no_resize, filedir = filedir)
        if image is not None:
          images.append(image)
        print('.', end='')
    print()
    if len(images):
      print(f'Extracting numbers from {len(images)} newspapers ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no pages numbers to extract for \'{self.newspaper_name}\'.')
    self.force = selfforce
    return images
  def get_head(self):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start extract pages head of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    images = []
    for page_pool in self:
      images.append(page_pool.get_head())
      print('.', end='')
    print()
    if len(images):
      print(f'Extracting head from {len(images)} newspapers ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no pages head to extract for \'{self.newspaper_name}\'.')
    self.force = selfforce
    return images
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
  def rename_folder(self, root):
    number = 1
    for filedir, dirs, files in os.walk(root):
      dirs.sort()
      for dir in dirs:
        if dir.isdigit():
          number = 1
          continue
        p = re.search(r'[^0-9]', dir).start()
        old_folder = os.path.join(filedir, dir)
        new_folder = os.path.join(filedir, dir[:p] + ' INS ' + str(number))
        if old_folder != new_folder:
          os.rename(old_folder, new_folder)
        number += 1






