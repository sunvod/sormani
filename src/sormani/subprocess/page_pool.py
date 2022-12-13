from __future__ import annotations

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
  def set_pages(self, pages):
    n_pages = len(self) if pages is None or pages > len(self) else pages
    if n_pages > 0:
      page = self[0]
      page.newspaper.set_n_pages(self, n_pages)
      self.sort(key=self._n_page_sort)
  def set_pages_already_seen(self, pages):
    n_pages = len(self) if pages is None or pages > len(self) else pages
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
  def get_pages_numbers(self, no_resize = False, filedir = None, pages = None):
    images = []
    if isinstance(pages, int):
      pages = [pages]
    for page in self:
      if pages is None or page.newspaper.n_page in pages:
        image = page.get_pages_numbers(no_resize=no_resize, filedir = filedir)
        if image is not None:
          images.append(image)
    return images
  def check_pages_numbers(self, model, save_images = False):
    errors = []
    countplusone = 0
    countminusone = 0
    countzero = 0
    for page in self:
      images, predictions = page.check_pages_numbers(model)
      if page.page_control == 0:
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
          ax[i // col][i % col].imshow(image[1])
          ax[i // col][i % col].set_title(title + ' ' + str(predictions[i]), fontsize = 7)
        plt.axis("off")
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
          n = 'X' if predictions[i] == 10 else str(predictions[i])
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
  def convert_image(self, page):
    try:
      image = Image.open(page.original_image)
      for convert in page.conversions:
        path_image = os.path.join(page.jpg_path, convert.image_path)
        Path(path_image).mkdir(parents=True, exist_ok=True)
        file = os.path.join(path_image, page.file_name) + '.jpg'
        if self.force or not Path(file).is_file():
          if image.size[0] < image.size[1]:
            wpercent = (convert.resolution / float(image.size[1]))
            xsize = int((float(image.size[0]) * float(wpercent)))
            image = image.resize((xsize, convert.resolution), Image.Resampling.LANCZOS)
          else:
            wpercent = (convert.resolution / float(image.size[0]))
            ysize = int((float(image.size[1]) * float(wpercent)))
            image = image.resize((convert.resolution, ysize), Image.Resampling.LANCZOS)
          image.save(file, 'JPEG', dpi=(convert.dpi, convert.dpi), quality=convert.quality)
    except Exception:
      tb = sys.exc_info()
      pass
  def convert_images(self, converts):
    if converts is None:
      return
    for page in self:
      page.add_conversion(converts)
    if len(self):
      start_time = time.time()
      print(f'Starting converting images of \'{self.newspaper_name}\' of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
      with Pool(processes=14) as mp_pool:
        mp_pool.map(self.convert_image, self)
        print(f'Conversion of {len(self)} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to convert for \'{self.newspaper_name}\'.')
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

