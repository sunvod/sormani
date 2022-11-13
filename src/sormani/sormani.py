import time
import datetime
import os

import tkinter as tk
from PIL import Image, ImageTk
from src.sormani.page import Images_group
import numpy as np


class Conversion:
  def __init__(self, image_path, dpi, quality, resolution):
    self.image_path = image_path
    self.dpi = dpi
    self.quality = quality
    self.resolution = resolution

class Sormani():
  def __init__(self,
               newspaper_name,
               root = '/mnt/storage01/sormani',
               year = None,
               month = None,
               day = None,
               ext = 'tif',
               image_path ='Tiff_images',
               path_exclude = [],
               path_exist ='pdf',
               force = False):
    self.newspaper_name = newspaper_name
    self.root = root
    self.i = 0
    self.elements = []
    root = os.path.join(root, image_path, newspaper_name)
    if not os.path.exists(root):
      print(f'{newspaper_name} non esiste in memoria.')
      return
    if year is not None:
      root = os.path.join(root, str(year))
      if not os.path.exists(root):
        print(f'{newspaper_name} per l\'anno {year} non esiste in memoria.')
        return
      if month is not None:
        root = os.path.join(root, str(month))
        if not os.path.exists(root):
          print(f'{newspaper_name} per l\'anno {year} e per il mese {month} non esiste in memoria.')
          return
        if day is not None:
          root = os.path.join(root, str(day))
          if not os.path.exists(root):
            print(f'{newspaper_name} per l\'anno {year}, per il mese {month} e per il giorno {day} non esiste in memoria.')
            return
    self.ext = ext
    self.image_path = image_path
    self.path_exclude = path_exclude
    self.path_exist = path_exist
    self.force = force
    self.converts = None
    self.elements = self.get_elements(root)
    if self.divide_image(self.elements):
      self.elements = self.get_elements(root)
    pass
  def get_elements(self, root):
    elements = []
    for filedir, dirs, files in os.walk(root):
      n_pages = len(files)
      if filedir in self.path_exclude or n_pages == 0:
        continue
      files.sort(key = self._get_elements)
      elements.append(Images_group(os.path.join(self.root, self.image_path, self.newspaper_name), self.newspaper_name, filedir, files, get_head = True))
    if len(elements) > 1:
      elements.sort(key=self._elements_sort)
    return elements
  def _get_elements(self, n):
    # n = e[:5]
    n = ''.join(c for c in n if c.isdigit())
    if not len(n):
      n = '0'
    r = ['0' for x in range(25 - len(n))]
    r = ''.join(r) + n
    return r
  def divide_image(self, elements):
    if not len(elements):
      return
    some_modify = False
    count = 0
    start_time = time.time()
    print(f'Starting dividing of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    for image_group in elements:
      for file_name in image_group.files:
        file_path = os.path.join(image_group.filedir, file_name)
        im = Image.open(file_path)
        width, height = im.size
        if width < height:
          continue
        some_modify = True
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
        count += 1
    if count:
      print(f'Dividing of {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'No dividing is necessary for \'{self.newspaper_name}\'.')
    return some_modify
  def __len__(self):
    return len(self.elements)
  def __iter__(self):
    return self
  def __next__(self):
    if self.i < len(self.elements):
      x = self.elements[self.i].get_page_pool(self.newspaper_name, self.root, self.ext, self.image_path,
                                              self.path_exist, self.force)
      self.i += 1
      return x
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
    pages = self.extract_pages(range = (3, 4))
    pages = int(pages[0]) + 1 if isinstance(pages, list) and len(pages) > 0 and pages[0].isdigit else None
    for page_pool in self:
      page_pool.create_pdf(pages)
      page_pool.set_files_name()
      page_pool.convert_images(converts)
  def change_all_contrasts(self, contrast = None, force = False):
    if not len(self.elements):
      return
    start_time = time.time()
    print(f'Starting changing contrasts of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = force
    for page_pool in self:
      count += page_pool.change_contrast(contrast, force)
    if count:
      print(f'Changing contrasts of {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no image to change contrast for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def create_jpg(self):
      for page_pool in self:
        page_pool.convert_images(converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)])
  def extract_pages(self, mute = True, image_mute = True, range = None):
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
      print(f'Starting extracting page number from \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    pages = []
    for page_pool in self:
      for i, page in enumerate(page_pool):
        if range is not None and i + 1 < range[0]:
          continue
        if range is not None and i + 1 >= range[1]:
          break
        page_number, images = page.extract_page()
        pages.append(page_number)
        count += 1
        if mute:
          continue
        if not i:
          page_number = 1
        print(f'{page.file_name} has page number: {page_number}')
        if not image_mute and page_number == '??' and images is not None:
          if isinstance(images, list):
            for image in images:
              image.show()
          else:
            images.show()
    if not mute:
      if count:
        print(f'Extracting page number from {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
      else:
        print(f'Warning: No extraction of page number are made for \'{self.newspaper_name}\'.')
    self.force = selfforce
    return pages

