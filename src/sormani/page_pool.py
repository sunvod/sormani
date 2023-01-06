from __future__ import annotations

import os
import pathlib
import time
import datetime
import cv2

from multiprocessing import Pool
from src.sormani.system import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Page_pool(list):
  def __init__(self, newspaper_name, date, force = False):
    self.newspaper_name = newspaper_name
    self.date = date
    self.force = force
  def set_pages(self):
    n_pages = len(self)
    if n_pages > 0:
      page = self[0]
      page.newspaper.set_n_pages(self, n_pages)
      self.sort(key=self._n_page_sort)
  def set_pages_already_seen(self):
    n_pages = len(self)
    for page in self:
      try:
        n_page = page.file_name.split('_')[-1][1:]
        page.newspaper.n_pages = n_pages
        page.newspaper.n_real_pages = len(self)
        page.newspaper.n_page = int(n_page)
      except:
        pass
  def _set_pages_sort(self, page):
    return page.original_file_name
    #return os.path.getmtime(Path(page.original_image))
  def _n_page_sort(self, page):
    return page.newspaper.n_page
  def isAlreadySeen(self):
    for page in self:
      return page.isAlreadySeen()
  def save_pages_images(self, storage):
    count = 0
    for page in self:
      if page.save_pages_images(storage):
        count += 1
    return count
  def get_pages_numbers(self, no_resize = False, filedir = None, pages = None, save_head = True):
    images = []
    if isinstance(pages, int):
      pages = [pages]
    for page in self:
      if pages is None or page.newspaper.n_page in pages:
        image = page.get_pages_numbers(no_resize=no_resize, filedir = filedir, save_head = save_head)
        if image is not None:
          images.append(image)
    return images
  def check_pages_numbers(self, model, save_images = False):
    errors = []
    countplusone = 0
    countminusone = 0
    countzero = 0
    for page in self:
      head_image, images, _, predictions = page.check_pages_numbers(model)
      if page.page_control == 0:
        if head_image is not None:
          plt.axis("off")
          plt.title(head_image[0])
          plt.imshow(head_image[1])
        countzero += 1
        errors.append(page.newspaper.n_page)
        col = 6
        row = len(images) // col + (1 if len(images) % col != 0 else 0)
        if row < 2:
          row = 2
        fig, ax = plt.subplots(row, col)
        for i in range(row):
          for j in range(col):
            ax[i][j].set_axis_off()
        title = str(self.date.strftime("%d/%m")) + ' ' + str(page.newspaper.n_page)
        for i, image in enumerate(images):
          if image is not None:
            ax[i // col][i % col].imshow(image[1])
            ax[i // col][i % col].set_title(title + ' ' + str(predictions[i]), fontsize = 7)
        plt.axis("off")
        # plt.title(head_image[0])
        plt.show()
      if save_images and images is not None and predictions is not None:
        if page.page_control == 1:
          countplusone += 1
          exact = 'sure'
        else:
          countminusone += 1
          exact = 'notsure'
        name = self.newspaper_name.lower().replace(' ', '_')
        Path(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, NO_NUMBERS)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, NUMBERS)).mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
          # n = 'X' if predictions[i] == 10 else str(predictions[i])
          n = page.newspaper.get_dictionary()[predictions[i]]
          if n == 'X':
            dir = NO_NUMBERS
          else:
            dir = NUMBERS
          file_name = image[0] + '_' + str(n)
          cv2.imwrite(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, dir, file_name) + '.jpg', image[1])
    if len(errors) < 2:
      print(f'{self.newspaper_name} del giorno {str(self.date.strftime("%d/%m/%y"))} ha le pagine esatte (code: {countminusone} {countzero} {countplusone}).')
    else:
      msg = '{} del giorno {} ha le pagine {} non esatte  (code: {} {} {}).'.format(self.newspaper_name, str(self.date.strftime("%d/%m/%y")), errors, countminusone, countzero, countplusone)
      print(msg)
      with portalocker.Lock('sormani_check.log', timeout=120) as sormani_log:
        sormani_log.write(msg + '\n')
  def get_head(self):
    images = []
    for page in self:
      images.append(page.get_head())
    return images
  def create_pdf(self, number = None, ocr = True):
    if len(self):
      self.number = number
      self.ocr = ocr
      start_time = time.time()
      print(f'Start creating pdf/a of \'{self.newspaper_name}\' of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.to_pdfa, self)
      print(f'The creation of {len(self)} pdf/a files for \'{self.newspaper_name}\' ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to process for \'{self.newspaper_name}\'.')
  def to_pdfa(self, page):
    _parser, options, plugin_manager = get_parser_options_plugins(None)
    options = Namespace()
    Path(os.path.join(page.pdf_path, 'pdf')).mkdir(parents=True, exist_ok=True)
    Path(page.txt_path).mkdir(parents=True, exist_ok=True)
    if self.ocr:
      exec_ocrmypdf(page.original_image, page.pdf_file_name, page.txt_file_name, ORIGINAL_DPI, UPSAMPLING_DPI)
    else:
      image = Image.open(page.original_image)
      image.save(page.pdf_file_name, "PDF", resolution=50.0)
      image.close()
    page.add_pdf_metadata(self.number)
  def set_image_file_name(self):
    for page in self:
      page.set_file_names()
      if page.original_file_name != page.file_name:
        new_file_name = os.path.join(page.original_path, page.file_name + pathlib.Path(page.original_image).suffix)
        if Path(page.original_image).is_file():
          os.rename(page.original_image, new_file_name)
          page.file_name = new_file_name
          page.newspaper.file_path = new_file_name
  def convert_images(self, converts):
    if converts is None:
      return
    for page in self:
      page.add_conversion(converts)
    if len(self):
      start_time = time.time()
      print(f'Starting converting images of \'{self.newspaper_name}\' of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.convert_image, self)
      print(f'Conversion of {len(self)} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to convert for \'{self.newspaper_name}\'.')
  def convert_image(self, page):
    page.convert_image(self.force)
  def extract_pages(self, range, mute = True, image_mute = True):
    pages = []
    for i, page in enumerate(self):
      if range is not None and i + 1 < range[0]:
        continue
      if range is not None and i + 1 >= range[1]:
        break
      page_number, images = page.extract_page()
      pages.append(page_number)
      if mute:
        continue
      if not i:
        page_number = 1
      print(f'{page.file_path} has page number: {page_number}')
      if not image_mute and page_number == '??' and images is not None:
        if isinstance(images, list):
          for image in images:
            image.show()
        else:
          images.show()
    return pages
  def rename_pages_files(self, model):
    file_to_be_changing = []
    end_flag = False
    next_page = -1
    while not end_flag:
      for page in self:
        if next_page > 1:
          next_page -= 1
          continue
        file_to_be_changing, end_flag, next_page = page.rename_pages_files(file_to_be_changing, model)
        if end_flag:
          break
        if next_page >= 0:
          break
      # if next_page < 0:
      #   end_flag = True
    flag = False
    filedirs = []
    for i, (old_file, new_file) in enumerate(file_to_be_changing):
      for j, (old_file2, new_file2) in enumerate(file_to_be_changing):
        if j <= i:
          continue
        if new_file == new_file2:
          flag = True
          print(new_file)
        path = os.path.dirname(new_file)
        if not path in filedirs:
          filedirs.append(path)
    if flag:
      raise OSError('Le modifiche fatte non sono valide')
    for dir in filedirs:
      filedir, dirs, files = next(os.walk(dir))
      files.sort()
      _, _, _files = next(os.walk(dir))
      new_files = []
      for (old_file, new_file) in file_to_be_changing:
        if os.path.basename(old_file) in files:
          files.remove(os.path.basename(old_file))
        new_files.append(os.path.basename(new_file))
      file_to_be_changed = []
      for file in files:
        if file in new_files:
          file_to_be_changed.append(file)
          print(file)
      if len(file_to_be_changed):
        raise OSError('Le modifiche fatte non sono valide perchè uno o più nuovi file sono già presenti')
    for old_file, new_file in file_to_be_changing:
      ext = pathlib.Path(old_file).suffix
      if ext == '.pdf':
        n = new_file.split('_')[-1][1:]
        if n.isdigit():
          n = int(n)
          page.newspaper.n_page = n
          page.add_pdf_metadata()
          page.add_jpg_metadata()
      os.rename(old_file, new_file + '.***')
    for dir in filedirs:
      for filedir, dirs, files in os.walk(dir):
        for file in files:
          ext = pathlib.Path(file).suffix
          if ext == '.***':
            new_file = Path(file).stem
            os.rename(os.path.join(filedir, file), os.path.join(filedir, new_file))
            date = datetime.datetime.now()
            modTime = time.mktime(date.timetuple())
            os.utime(os.path.join(filedir, new_file), (modTime, modTime))
  def update_date_creation(self):
    if not len(self):
      return
    for page in self:
      date = datetime.datetime.now()
      modTime = time.mktime(date.timetuple())
      for filedir, _, files in os.walk(page.original_path):
        for file in files:
          os.utime(os.path.join(filedir, file), (modTime, modTime))
      if os.path.isdir(page.pdf_path):
        for filedir, _, files in os.walk(page.pdf_path):
          for file in files:
            os.utime(os.path.join(os.path.join(filedir, file)), (modTime, modTime))
      return
  def check_jpg(self, converts, integrate=False):
    if converts is None:
      return
    if not len(self):
      return
    for page in self:
      jpg_path = page.jpg_path
      break
    if not Path(os.path.join(jpg_path, 'pdf')).is_dir():
      type = jpg_path.split('/')[-1]
      print(f'{self.newspaper_name} del giorno {str(self.date.strftime("%d/%m/%Y"))} di tipo \'{type}\' non ha il pdf.')
      return
    _, dirs, files = next(os.walk(os.path.join(jpg_path, 'pdf')))
    file_count = len(files)
    if file_count:
      for convert in converts:
        exist = Path(os.path.join(jpg_path, convert.image_path)).is_dir()
        if exist:
          _, _, files = next(os.walk(os.path.join(jpg_path, convert.image_path)))
        if not exist or file_count != len(files):
          type = jpg_path.split('/')[-1]
          print(f'{self.newspaper_name} del giorno {str(self.date.strftime("%d/%m/%Y"))} di tipo \'{type}\' non ha il jpg di tipo {convert.image_path} con dpi={convert.dpi}')
          if integrate:
            self.convert_images([convert])
