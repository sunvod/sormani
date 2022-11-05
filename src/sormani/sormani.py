
import os
import time
import datetime
import os

from src.sormani.page import Images_group


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
    for filedir, dirs, files in os.walk(root):
      n_pages = len(files)
      if filedir in path_exclude or n_pages == 0:
        continue
      files.sort()
      self.elements.append(Images_group(filedir, files))
    self.elements.sort(key = self._elements_sort)
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
      raise StopIteration
  def _elements_sort(self, images_group):
    return images_group.date
  def set_force(self, force):
    self.force = force
  def create_all_images(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)]):
    for page_pool in self:
      page_pool.create_pdf()
      page_pool.set_files_name()
      page_pool.convert_images(converts)
  def change_all_contrasts(self, contrast = 50):
    start_time = time.time()
    print(f'Starting converting images of \'{self.newspaper_name}\' at {str(datetime.datetime.now())}')
    for page_pool in self:
      page_pool.change_contrast(contrast)
    if len(self):
      print(f'Conversion of {len(self)} images ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to convert for \'{self.newspaper_name}\'.')
  def create_jpg(self):
      for page_pool in self:
        page_pool.convert_images(converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)])

