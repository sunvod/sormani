import multiprocessing
import time
import datetime
import pathlib
import re
import os
import cv2
import numpy as np
from PIL.Image import DecompressionBombError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

from multiprocessing import Pool

from src.sormani.image_group import Images_group
from src.sormani.system import *

import warnings
warnings.filterwarnings("ignore")


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
               exclude_ins=False,
               only_ins=False,
               valid_ins=[],
               notcheckimages=True,
               thresholding=0,
               rename_folders=True,
               model_path=None,
               use_ai=False):
    if year is not None and isinstance(year, list) and not len(year):
      error = 'Non è stato indicato l\'anno di estrazione. L\'esecuzione terminerà.'
      raise OSError(error)
    if months is not None and isinstance(months, list) and not len(months):
      error = 'Non è stato indicato il mese di estrazione. L\'esecuzione terminerà.'
      raise OSError(error)
    if days is not None and isinstance(days, list) and not len(days):
      error = 'Non è stato indicato il giorno di estrazione. L\'esecuzione terminerà.'
      raise OSError(error)
    self.thresholding = thresholding
    if not isinstance(newspaper_names, list):
      if newspaper_names is not None:
        name = newspaper_names
        newspaper_names = []
        newspaper_names.append(name)
      else:
        newspaper_names = NEWSPAPERS
        rename_only = True
    if not isinstance(months, list):
      months = [months]
    months.sort()
    if not isinstance(days, list):
      days = [days]
    days.sort()
    self.days = days
    self.dir_name = ''
    self.roots = []
    if model_path is not None and use_ai:
      self.set_GPUs()
    for newspaper_name in newspaper_names:
      for month in months:
        for day in days:
          self._init(newspaper_name,
                     root,
                     year,
                     month,
                     day,
                     ext,
                     image_path,
                     path_exclude,
                     path_exist,
                     force,
                     exclude_ins,
                     only_ins,
                     valid_ins,
                     notcheckimages,
                     rename_folders,
                     model_path,
                     use_ai)
    self.set_elements()
    self.pages_pool = []
    for _ in self:
      pass
  def _init(self,
            newspaper_name,
            root,
            year,
            month,
            day,
            ext,
            image_path,
            path_exclude,
            path_exist,
            force,
            exclude_ins,
            only_ins,
            valid_ins,
            notcheckimages,
            rename_folders,
            model_path,
            use_ai):
    self.newspaper_name = newspaper_name
    self.root = root
    self.i = 0
    self.elements = []
    self.day = day
    new_root = os.path.join(root, image_path, newspaper_name)
    self.add_zero_to_dir(new_root)
    if not os.path.exists(new_root):
      print(f'{newspaper_name} non esiste in memoria.')
      return self.elements
    if year is not None:
      new_root = os.path.join(new_root, str(year))
      self.complete_root = new_root
      if month is not None:
        new_root = os.path.join(new_root, self.add_zero(month))
        self.complete_root = new_root
        if day is not None:
          new_root = os.path.join(new_root, self.add_zero(day))
          self.complete_root = new_root
    self.dir_name = self.complete_root.split('/')[-1] if self.dir_name == '' else self.dir_name + ',' + self.complete_root.split('/')[-1]
    self.new_root = new_root
    if rename_folders:
      self.rename_folders()
    self.ext = ext
    self.image_path = image_path
    self.path_exclude = path_exclude
    self.path_exist = path_exist
    self.force = force
    self.converts = None
    self.exclude_ins = exclude_ins
    self.only_ins = only_ins
    self.valid_ins = valid_ins
    self.notcheckimages = notcheckimages
    self.roots.append(self.new_root)
    self.model = None
    if model_path is not None and use_ai:
      model_path = os.path.join('models', self.newspaper_name.lower().replace(' ', '_'), model_path)
      self.model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, model_path))
    self.model_path = model_path
    self.use_ai = use_ai
  def __len__(self):
    return len(self.elements)
  def __iter__(self):
    return self
  def __next__(self):
    if self.i < len(self.elements):
      if self.i >= len(self.pages_pool) or self.pages_pool[self.i] is None:
        # if self.model is not None and not self.i:
        #   self.set_GPUs()
        page_pool = self.elements[self.i].get_page_pool(self.newspaper_name,
                                                        self.new_root,
                                                        self.ext,
                                                        self.image_path,
                                                        self.path_exist,
                                                        self.force,
                                                        self.thresholding,
                                                        self.model,
                                                        self.use_ai)
        if len(page_pool):
          if page_pool.isAlreadySeen():
            page_pool.set_pages_already_seen()
          else:
            page_pool.set_pages(use_ai=self.use_ai)
        if self.i < len(self.pages_pool) and self.pages_pool[self.i] is None:
          self.pages_pool[self.i] = page_pool
        else:
          self.pages_pool.append(page_pool)
      else:
        page_pool = self.pages_pool[self.i]
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
    page_pool = self.elements[self.i].get_page_pool(
      self.newspaper_name, self.dir_name, self.ext, self.image_path, self.path_exist, self.force)
    page_pool.set_pages()
    self.i += 1
    return page_pool[0].newspaper.number
  def add_zero(self, n):
    if isinstance(n, int):
      n = str(n)
    if n.isdigit() and len(n) == 1:
      n = '0' + n
    return n
  def add_zero_to_dir(self, root):
    to_repeat = False
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
            to_repeat = True
          except:
            pass
    if to_repeat:
      self.add_zero_to_dir(root)
  def set_elements(self):
    self.elements = []
    _filedirs = []
    for root in self.roots:
      filedirs = []
      roots = []
      if self.days[0] is not None:
        new_root = '/'.join(root.split('/')[:-1])
        for filedir, dirs, files in os.walk(new_root):
          if not len(dirs):
            roots.append(root)
          else:
            dirs.sort()
            for dir in dirs:
              n = dir.split(' ')[0]
              if n.isdigit():
                if int(n) in self.days:
                  roots.append(os.path.join(new_root, dir))
            break
      else:
        roots.append(root)
      roots.sort()
      for root in roots:
        for filedir, dirs, files in os.walk(root):
          dir = filedir.split('/')[-1]
          if filedir in self.path_exclude or \
              len(files) == 0 or \
              len(dirs) > 0 or \
              (self.exclude_ins and not dir.isdigit()) or \
              (self.only_ins and dir.isdigit()):
            continue
          if isinstance(self.valid_ins, int):
            self.valid_ins = [self.valid_ins]
          if len(dir.split()) >= 2 and dir.split()[1] == 'INS':
            valid_ins = [str(x) for x in self.valid_ins]
            if not dir.split()[2] in valid_ins:
              continue
          files.sort(key = self._get_elements)
          filedirs.append((filedir, files))
      filedirs.sort()
      for filedir, files in filedirs:
        if self.notcheckimages or self.check_if_image(filedir, files):
          if not filedir in _filedirs:
            self.elements.append(Images_group(os.path.join(self.root, self.image_path, self.newspaper_name),
                                              self.newspaper_name,
                                              filedir,
                                              files))
            _filedirs.append(filedir)
  def check_if_image(self, filedir, files):
    for file_name in files:
      try:
        Image.open(os.path.join(filedir, file_name))
      except DecompressionBombError:
        pass
      except:
        with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
          sormani_log.write('No valid Image: ' + os.path.join(filedir, file_name) + '\n')
        print(f'Not a valid image: {os.path.join(filedir, file_name)}')
        return False
      img = cv2.imread(os.path.join(filedir, file_name), cv2.IMREAD_UNCHANGED)
      if img.shape[2] == 4:
        img = img[:,:,:3]
        cv2.imwrite(os.path.join(filedir, file_name), img)
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
  def create_all_images(self,
                        ocr = True,
                        converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)],
                        number = None,
                        thresholding=None):
    if not len(self.elements):
      return
    for page_pool in self:
      if not len(page_pool):
        continue
      if thresholding is not None:
        page_pool.thresholding = thresholding
      page_pool.create_pdf(number, ocr = ocr)
      page_pool.convert_images(converts)
  def convert_all_images(self,
                        ocr = True,
                        converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)],
                        number = None):
    if not len(self.elements):
      return
    selfforce = self.force
    self.force = True
    for page_pool in self:
      if not len(page_pool):
        continue
      page_pool.convert_images(converts)
    self.force = selfforce
  def set_all_images_names(self, force_rename=False):
    if not len(self.elements):
      return
    for page_pool in self:
      if not len(page_pool):
        continue
      else:
        if not page_pool.isAlreadySeen():
          page_pool.set_image_file_name()
          i = self.pages_pool.index(page_pool)
          self.pages_pool[i] = None
        if force_rename:
          s = page_pool.filedir.split('/')[-1]
          s = s.split()[-1]
          n_page = 0
          if s[0] == 'P':
            if s[1:].isdigit():
              n_page = int(s[1:])
          if s[0:2] == 'OT':
            s = s.split('-')[-1]
            if s.isdigit():
              n_page = int(s)
          if n_page:
            page_pool.force_rename_image_file_name(n_page=n_page)
            i = self.pages_pool.index(page_pool)
            self.pages_pool[i] = None
        page_pool.set_pages_already_seen()
    self.set_elements()
  def change_contrast(self, contrast = None):
    if not len(self.elements):
      return
    selfforce = self.force
    self.contrast = contrast
    self.force = True
    for page_pool in self:
      page_pool.change_contrast(contrast=self.contrast, force=self.force)
    self.force = selfforce
  def change_threshold(self, limit = None, color = 255, inversion=False):
    if not len(self.elements):
      return
    selfforce = self.force
    global global_count_contrast
    global_count_contrast.value = 0
    self.limit = limit
    self.color = color
    self.inversion = inversion
    self.force = True
    for page_pool in self:
      page_pool.change_threshold(limit=limit, color=color, inversion=inversion)
    self.force = selfforce
  def change_colors(self, limit=None, color=255, inversion=False):
    if not len(self.elements):
      return
    selfforce = self.force
    global global_count_contrast
    global_count_contrast.value = 0
    self.limit = limit
    self.color = color
    self.inversion = inversion
    self.force = True
    for page_pool in self:
      page_pool.change_colors(limit=limit, color=color, inversion=inversion)
    self.force = selfforce
  def improve_images(self, limit=None, color=255, inversion=False, threshold="b9", debug=False):
    if not len(self.elements):
      return
    selfforce = self.force
    self.inversion = inversion
    self.force = True
    for page_pool in self:
      page_pool.improve_images(limit=limit, color=color, inversion=inversion, threshold=threshold, debug=debug)
    self.force = selfforce
  def clean_images(self, limit=None, color=255, inversion=False, threshold="b9", debug=False):
    if not len(self.elements):
      return
    selfforce = self.force
    self.inversion = inversion
    self.force = True
    for page_pool in self:
      page_pool.clean_images(limit=limit, color=color, inversion=inversion, threshold=threshold, debug=debug)
    self.force = selfforce
  def divide_image(self, is_bobina = False):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(f'Starting division of \'{self.newspaper_name}\' in date {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    # with Pool(processes=N_PROCESSES_SHORT) as mp_pool:
    #   mp_pool.map(self.divide_image, self.elements)
    count = 0
    for page_pool in self:
      n = page_pool.divide_image(is_bobina)
      if n:
        count += n
        i = self.pages_pool.index(page_pool)
        self.pages_pool[i] = None
    if count:
      # print()
      print(
        f'Division of {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
      self.set_elements()
    else:
      print(f'No division is needed for \'{self.newspaper_name}\'.')
  def remove_borders(self, limit = 5000, verbose = False):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(f'Starting removing borders of \'{self.newspaper_name}\' in date {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    for page_pool in self:
      count = page_pool.remove_borders(limit = limit, verbose = verbose)
    if count:
      print(
        f'Removing borders of {count} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
      self.set_elements()
    else:
      print(f'No removing borders is needed for \'{self.newspaper_name}\'.')
  def add_pdf_metadata(self, first_number = None):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start redefine Metadata of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = True
    self.first_number = first_number
    for page_pool in self:
      count = page_pool.add_pdf_metadata(first_number = first_number)
    if count:
      print(f'Redefinition Metadata of {sum(count)} pages ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no Metadata to be redefined for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def add_jpg_metadata(self, first_number = None):
    if not len(self.elements):
      return
    global global_count
    global_count.value = 0
    start_time = time.time()
    print(
      f'Start redefine Metadata of \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = True
    self.first_number = first_number
    # with Pool(processes=N_PROCESSES_SHORT) as mp_pool:
    #   mp_pool.map(self._add_pdf_metadata, self)
    for page_pool in self:
      self._add_jpg_metadata(page_pool)
    if global_count.value:
      print(f'It has redefined Metadata of {global_count.value} ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'There are no Metadata to be redefined for \'{self.newspaper_name}\'.')
    self.force = selfforce
  def _add_jpg_metadata(self, page_pool):
    count = 0
    for page in page_pool:
      page.add_jpg_metadata(self.first_number)
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
      print(f'Starting extracting page number from \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
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
      f'Start extract pages images of \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
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
  def get_pages_numbers(self, no_resize = False, filedir = None, pages = None, save_head = True, force=False, debug=False):
    if not len(self.elements):
      return
    if filedir is not None:
      filedir += '_' + self.newspaper_name.lower().replace(' ', '_')
    start_time = time.time()
    print(
      f'Start extract pages numbers of \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = True
    images = []
    for page_pool in self:
      # if not page_pool.isins:
      image = page_pool.get_pages_numbers(no_resize = no_resize, filedir = filedir, pages = pages, save_head = save_head, force=force, debug=debug)
      if image is not None and len(image):
        images.append(image)
      print('.', end='')
    print()
    self.force = selfforce
    return images
  def get_crop(self, no_resize = False, filedir = None, pages = None, force=False):
    if not len(self.elements):
      return
    if filedir is not None:
      filedir += '_' + self.newspaper_name.lower().replace(' ', '_')
    start_time = time.time()
    print(
      f'Start extract cropped images of \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    selfforce = self.force
    self.force = True
    images = []
    for page_pool in self:
      if not page_pool.isins:
        image = page_pool.get_crop(no_resize = no_resize, filedir = filedir, pages = pages, force=force)
        if image is not None and len(image):
          images.append(image)
        print('.', end='')
    print()
    self.force = selfforce
    return images
  def get_head(self):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start extract pages head of \'{self.newspaper_name}\' ({self.dir_name}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
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
  def rename_folders(self):
    number = 1
    for one_root in [self.new_root]:
      for filedir, dirs, files in os.walk(one_root):
        dirs.sort()
        for dir in dirs:
          if dir.isdigit() or not dir[:2].isdigit():
            number = 1
            continue
          p = re.search(r'[^0-9]', dir).start()
          old_folder = os.path.join(filedir, dir)
          new_folder = os.path.join(filedir, dir[:p] + ' INS ' + str(number))
          s = old_folder.split(' ')[-1]
          if s[0] == 'P' or s[0:2] == 'OT':
            new_folder += ' ' + old_folder.split(' ')[-1]
          if old_folder != new_folder:
            try:
              os.rename(old_folder, new_folder)
              pass
            except:
              error = 'La directory \'' + new_folder + '\' non può essere modificata (probabilmente non è vuota).'
              raise OSError(error)
          number += 1
  def set_GPUs(self):
    try:
      from numba import cuda
      device = cuda.get_current_device()
      device.reset()
      gpus = tf.config.list_physical_devices('GPU')
      if gpus:
        try:  # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:  # Memory growth must be set before GPUs have been initialized
          print(e)
          exit(0)
    except:
      pass
  def check_page_numbers(self,
                         save_images = False,
                         print_images=True,
                         exclude_ins = False,
                         only_ins = False):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start check pages numbers of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    images = []
    if self.model is None:
      print("No artificial intelligence model is present and therefore no page control is possible.")
      return
    for page_pool in self:
      if (exclude_ins and page_pool.isins) or (only_ins and not page_pool.isins):
        continue
      page_pool.check_pages_numbers(self.model, save_images = save_images, print_images=print_images)
    if len(images):
      print(f'Checking numbers ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    self.force = selfforce
    return images
  def rename_pages_files(self, model_path = 'best_model_DenseNet201', assume_newspaper = False, do_prediction = False):
    if not len(self.elements):
      return
    selfforce = self.force
    self.force = True
    if assume_newspaper:
      model_path = os.path.join('models', self.newspaper_name.lower().replace(' ', '_'), model_path)
    else:
      model_path = os.path.join('models', model_path)
    model = None
    if do_prediction:
      model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, model_path))
    for page_pool in self:
      page_pool.rename_pages_files(model)
    self.force = selfforce
  def update_date_creation(self):
    if not len(self.elements):
      return
    start_time = time.time()
    print(
      f'Start update data creation of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    count = 0
    selfforce = self.force
    self.force = True
    images = []
    for page_pool in self:
      page_pool.update_date_creation()
    print()
    print(f'Update date creation ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    self.force = selfforce
    return images
  def check_jpg(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)], integrate=False):
    if not len(self.elements):
      return
    selfforce = self.force
    self.force = True
    start_time = time.time()
    print(f'Starting checking pdf at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    for page_pool in self:
      if not len(page_pool):
        continue
      page_pool.check_jpg(converts, integrate)
    self.force = selfforce
    print(f'Checking pdf ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    # print(f'Warning: There is no files to check for \'{self.newspaper_name}\'.')
  def set_bobine_merge_images(self):
    start_time = time.time()
    print(f'Starting merging images of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    count = 0
    for page_pool in self:
      count += page_pool.set_bobine_merge_images()
    self.set_elements()
    print(f'Merging {count} frames ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
  def set_bobine_select_images(self, remove_merge=True, write_borders=False, threshold = 5):
    start_time = time.time()
    print(f'Extracting frames of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    count = 0
    for page_pool in self:
      count += page_pool.set_bobine_select_images(remove_merge, write_borders, threshold)
    self.set_elements()
    print(f'Extracting {count} frames at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
  def rotate_fotogrammi(self, verbose=False, limit=5000):
    start_time = time.time()
    print(f'Start Rotate frames of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    count = 0
    for page_pool in self:
      count += page_pool.rotate_fotogrammi(verbose, limit)
    print(f'End Rotate {count} frames at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
  def set_fotogrammi_folders(self, model_path = 'best_model_DenseNet201'):
    start_time = time.time()
    print(f'Start set fotogrammi folders of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
    count = 0
    for page_pool in self:
      count += page_pool.set_fotogrammi_folders(model_path)
    print(f'Set  {count} fotogrammi folders at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
  def set_bobine_pipeline(self, no_set_names = False):
    self.set_bobine_merge_images()
    self.set_bobine_select_images(threshold=5)
    self.rotate_fotogrammi(limit=4000)
    self.remove_borders()
  def set_giornali_pipeline(self, divide = True, rename = True, change_contrast = True, create_images=True, force_rename=False):
    try:
      selfforce = self.force
    except:
      return
    self.force = True
    if divide:
      self.divide_image()
    if change_contrast:
      self.change_contrast()
    if rename:
      self.set_all_images_names(force_rename=force_rename)
    self.force = selfforce
    self.set_elements()
    if create_images:
      self.create_all_images()