from __future__ import annotations

import multiprocessing
import os
import pathlib
import time
import datetime
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity
import scipy.ndimage as ndi
import imagehash

from multiprocessing import Pool, Manager
from src.sormani.system import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# global_count = multiprocessing.Value('I', 0)
global_count_contrast = multiprocessing.Value('I', 0)

class Page_pool(list):
  def __init__(self, newspaper_name, filedir, name_complete, new_root, date, force = False, thresholding=None, ais=None):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.new_root = new_root
    self.name_complete = name_complete
    self.date = date
    self.force = force
    self.thresholding = thresholding if thresholding is not None else THRESHOLDING
    self.isOT = filedir.split(' ')[-1][0:2] == 'OT'
    self.need_rotation = filedir.split(' ')[-1][0:2] == 'OT' and len(filedir.split(' ')[-1].split('-')) > 1 and filedir.split(' ')[-1].split('-')[1].isdigit()
    self.ais = ais

  def _n_page_sort(self, page):
    n_page = str(page.newspaper.n_page)
    if n_page[0] != '?':
      n_page = ('000' + n_page)[-3:]
    return n_page
  def set_pages(self, use_ai=False):
    n_pages = len(self)
    for i, page in enumerate(self):
      if not i:
        page.newspaper.isfirstpage = True
      else:
        page.newspaper.isfirstpage = False
    if n_pages > 0:
      page = self[0]
      page.newspaper.set_n_pages(self, n_pages, use_ai)
      self.sort(key=self._n_page_sort)
  def set_pages_already_seen(self):
    n_pages = len(self)
    for i, page in enumerate(self):
      if not i:
        page.newspaper.isfirstpage = True
      else:
        page.newspaper.isfirstpage = False
      n_page = page.file_name.split('_')[-1][1:]
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(self)
      if n_page.isdigit():
        page.newspaper.n_page = int(n_page)
      else:
        page.newspaper.n_page = n_page
  def _set_pages_sort(self, page):
    return page.original_file_name
  def isAlreadySeen(self):
    for page in self:
      return page.isAlreadySeen()
  def save_pages_images(self, storage):
    count = 0
    for page in self:
      if page.save_pages_images(storage):
        count += 1
    return count
  def get_pages_numbers(self, no_resize = False, filedir = None, pages = None, save_head = True, force=False, debug=False):
    images = []
    if isinstance(pages, int):
      pages = [pages]
    for page in self:
      page.isins = self.isins
      if pages is None or page.newspaper.n_page in pages:
        page.debug = debug
        image = page.get_pages_numbers(no_resize=no_resize, filedir = filedir, save_head = save_head, force=force)
        if image is not None:
          images.append(image)
    return images
  def get_crop(self, no_resize = False, filedir = None, force=False):
    for page in self:
      page.no_resize = False
      page.filedir = None
      page.force = False
      with Pool(processes=N_PROCESSES) as mp_pool:
        images = mp_pool.map(self._get_crop, self)
    return images
  def _get_crop(self, page):
    return page.get_crop()
  def check_pages_numbers(self, model, save_images = False, print_images=True):
    if self.date is None:
      return
    errors = []
    countplusone = 0
    countminusone = 0
    countzero = 0
    found_qm = False
    for page in self:
      page.isins = self.isins
      head_images, images, _, predictions, isvalid = page.check_pages_numbers(model)
      if not isvalid:
        msg = '{} ({}) del giorno {} non ha i nomi dei file con le date specificate.'.format(self.newspaper_name, self.name_complete, str(self.date.strftime("%d/%m/%y")))
        print(msg)
        with portalocker.Lock('sormani_check.log', timeout=120) as sormani_log:
          sormani_log.write(msg + '\n')
        return
      if page.page_control == 0:
        if head_images is not None and print_images:
          for head_image in head_images:
            plt.axis("off")
            plt.title(head_image[0])
            plt.imshow(head_image[1])
            plt.show()
        countminusone += 1
        errors.append(page.newspaper.n_page)
        if print_images:
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
          plt.show()
      elif page.page_control == 1:
        countplusone += 1
      elif page.page_control == 2:
        errors.append(page.newspaper.n_page)
        found_qm = True
        countminusone += 1
      else:
        countzero += 1
      if save_images and images is not None and predictions is not None:
        if page.page_control == 1:
          exact = 'sure'
        else:
          exact = 'notsure'
        name = self.newspaper_name.lower().replace(' ', '_')
        Path(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, NO_NUMBERS)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, NUMBERS)).mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
          n = page.newspaper.get_dictionary()[predictions[i]]
          if n == 'X':
            dir = NO_NUMBERS
          else:
            dir = NUMBERS
          file_name = image[0] + '_' + str(n)
          cv2.imwrite(os.path.join(STORAGE_BASE, REPOSITORY, name, exact, dir, file_name) + '.jpg', image[1])
    if ((not found_qm and len(errors) < 2) or not len(errors)) and countzero <= countplusone // 4:
      if countzero > 1:
        print(f'{self.newspaper_name} ({self.name_complete}) del giorno {str(self.date.strftime("%d/%m/%y"))} ha le pagine esatte e ha {countzero} pagine indefinite (code: {countminusone} {countzero} {countplusone}).')
      elif countzero == 1:
          print(f'{self.newspaper_name} ({self.name_complete}) del giorno {str(self.date.strftime("%d/%m/%y"))} ha le pagine esatte e ha {countzero} pagina indefinita (code: {countminusone} {countzero} {countplusone}).')
      else:
        print(f'{self.newspaper_name} ({self.name_complete}) del giorno {str(self.date.strftime("%d/%m/%y"))} ha le pagine esatte (code: {countminusone} {countzero} {countplusone}).')
    else:
      if len(errors) == 1:
        msg = '{} ({}) del giorno {} ha la pagina {} non esatta e ha {} pagine indefinite (code: {} {} {}).'.format(
          self.newspaper_name,
          self.name_complete, str(self.date.strftime("%d/%m/%y")), errors, countzero, countminusone, countzero, countplusone)
      elif len(errors) > 1:
        msg = '{} ({}) del giorno {} ha le pagine {} non esatte e ha {} pagine indefinite (code: {} {} {}).'.format(
          self.newspaper_name, self.name_complete, str(self.date.strftime("%d/%m/%y")), errors, countzero, countminusone, countzero, countplusone)
      else:
        msg = '{} ({}) del giorno {} ha {} pagine indefinite (code: {} {} {}).'.format(
          self.newspaper_name, self.name_complete, str(self.date.strftime("%d/%m/%y")), countzero, countminusone, countzero, countplusone)
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
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start creating pdf/a of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(
          f'Start creating pdf/a of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      if MULTIPROCESSING:
        with Pool(processes=N_PROCESSES) as mp_pool:
          mp_pool.map(self.to_pdfa, self)
      else:
        for page in self:
          self.to_pdfa(page)
      print(f'The creation of {len(self)} pdf/a files for of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to process for \'{self.newspaper_name}\'.')
  def to_pdfa(self, page):
    Path(os.path.join(page.pdf_path, 'pdf')).mkdir(parents=True, exist_ok=True)
    Path(page.txt_path).mkdir(parents=True, exist_ok=True)
    if self.ocr:
      try:
        exec_ocrmypdf(page.original_image,
                      page.pdf_file_name,
                      page.txt_file_name,
                      ORIGINAL_DPI,
                      UPSAMPLING_DPI,
                      thresholding=self.thresholding)
      except Exception as e:
        print(e)
        print(page.original_image)
    else:
      image = Image.open(page.original_image)
      try:
        image.save(page.pdf_file_name, "PDF", resolution=50.0)
      except Exception as e:
        print(e)
        pass
      image.close()
    page.add_pdf_metadata(self.number)
  def change_contrast(self, contrast = 50, force = True, number = None, ocr = True):
    if len(self):
      self.number = number
      self.ocr = ocr
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start changing the contrast of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(
          f'Start changing the contrast of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.contrast = contrast
        page.force = force
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self._change_contrast, self)
      print(f'The {len(self)} pages contrast change of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to changing the contrast for \'{self.newspaper_name}\'.')
  def _change_contrast(self, page):
    global global_count_contrast
    i = page.change_contrast()
    with global_count_contrast.get_lock():
      global_count_contrast.value += i
  def change_threshold(self, limit = 50, color = 255, inversion = False):
    if len(self):
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start changing the threshold of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(f'Start changing the threshold of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.limit = limit
        page.color = color
        page.inversion = inversion
      with Pool(processes=N_PROCESSES) as mp_pool:
        count = mp_pool.map(self._change_threshold, self)
      print(f'The {len(count)} pages threshold change of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to changing the threshold for \'{self.newspaper_name}\'.')
  def _change_threshold(self, page):
    return page.change_threshold()
  def change_colors(self, limit = 50, color = 255, inversion = False):
    if len(self):
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start changing the colors of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(f'Start changing the colors of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.limit = limit
        page.color = color
        page.inversion = inversion
      with Pool(processes=N_PROCESSES) as mp_pool:
        count = mp_pool.map(self._change_colors, self)
      print(f'The {len(count)} pages colors change of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to changing colors for \'{self.newspaper_name}\'.')
  def _change_colors(self, page):
    return page.change_colors()
  def improve_images(self, limit = 50, color = 255, inversion = False, threshold="b9", debug=False):
    if len(self):
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start improving images of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(f'Start improving images of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.limit = limit
        page.color = color
        page.inversion = inversion
        page.threshold = threshold
        page.debug = debug
      with Pool(processes=N_PROCESSES) as mp_pool:
        count = mp_pool.map(self._improve_images, self)
      print(f'The {len(count)} improved images of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to improve images for \'{self.newspaper_name}\'.')
  def _improve_images(self, page):
    return page.improve_images()
  def clean_images(self, color=248, threshold=230, final_threshold=200, last_threshold=160, use_ai=False):
    if len(self):
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      if self.date is not None:
        print(f'Start cleaning images of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(f'Start cleaning images of \'{self.newspaper_name}\' ({dir_name}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.color = color
        page.threshold = threshold
        page.final_threshold = final_threshold
        page.last_threshold = last_threshold
        page.valid = None
        page.use_ai = use_ai
      with Pool(processes=N_PROCESSES) as mp_pool:
        count = mp_pool.map(self._clean_images, self)
        count = sum(count)
      print(f'The {count} cleaned images of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to clean images for \'{self.newspaper_name}\'.')
  def _clean_images(self, page):
    return page.clean_images()
  def add_pdf_metadata(self, first_number = None):
    count = 0
    for page in self:
      page.first_number = first_number
    # for page in self:
    #   count += page.add_pdf_metadata()
    with Pool(processes=N_PROCESSES) as mp_pool:
      count = mp_pool.map(self._add_pdf_metadata, self)
    return count
  def _add_pdf_metadata(self, page):
    return page.add_pdf_metadata()
  def divide_image(self, is_bobina = False):
    flag = False
    for page in self:
      image = Image.open(page.original_image)
      width, height = image.size
      if width > height:
        flag = True
        break
    if flag:
      for page in self:
        last = Path(page.original_image).stem.split('_')[-1]
        if last == '0' or last == '1' or last == '2':
          error = '\'' + page.original_image + '\' into folder \'' + self.filedir + '\' seems to be inconsistent because images into the folder are only partially divided.' \
                                                                                 '\nIt is necessary reload all the folder from backup data in order to have consistency again.'
          raise ValueError(error)
    pages = []
    for page in self:
      page.is_bobina = is_bobina
      page.filedir = self.filedir
      file_name_no_ext = Path(page.original_image).stem
      file_path_no_ext = os.path.join(self.filedir, file_name_no_ext)
      ext = Path(page.original_image).suffix
      image = Image.open(page.original_image)
      width, height = image.size
      if self.need_rotation:
        img = cv2.imread(page.original_image)
        img = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(page.original_image, img)
      if width < height:
        if self.isOT and height > 8000:
          img = cv2.imread(page.original_image)
          img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
          cv2.imwrite(page.original_image, img)
        elif flag:
          os.rename(page.original_image, file_path_no_ext + '_0' + ext)
          page.original_image = file_path_no_ext + '_0' + ext
          continue
        else:
          continue
      pages.append(page)
    result = 0
    # if self.model is not None and self.use_ai:
    #   for page in pages:
    #     result += self._divide_image(page)
    # else:
    #   with Pool(processes=N_PROCESSES) as mp_pool:
    #     result = mp_pool.map(self._divide_image, pages)
    if MULTIPROCESSING:
      with Pool(processes=N_PROCESSES) as mp_pool:
        result = mp_pool.map(self._divide_image, pages)
        result = sum(result)
      return result
    else:
      count = 0
      for page in self:
        count += self._divide_image(page)
      return count
  def _divide_image(self, page):
    return page.divide_image()
  def remove_borders(self, limit = 5000, threshold=200):
    for page in self:
      page.limit = limit
      page.threshold = threshold
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._remove_borders, self)
    return sum(result)
  def _remove_borders(self, page):
    return page.remove_borders()
  def set_greyscale(self,):
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._set_greyscale, self)
    return sum(result)
  def _set_greyscale(self, page):
    return page.set_greyscale()
  def remove_frames(self, limit = 5000, threshold=180, default_frame=(0,0,0,0)):
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.default_frame = default_frame
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._remove_frames, self)
    if result is not None and all(v is not None for v in result):
      return sum(result)
    return 0
  def _remove_frames(self, page):
    return page.remove_frames()
  def remove_single_frames(self, limit = 5000, threshold=200, default_frame=(0,0,0,0), valid=[True,True,True,True]):
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.default_frame = default_frame
      page.valid = valid
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._remove_single_frames, self)
    if result is not None and all(v is not None for v in result):
      return sum(result)
    return 0
  def _remove_single_frames(self, page):
    return page.remove_single_frames()
  def remove_last_single_frames(self, limit, threshold, default_frame):
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.default_frame = default_frame
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._remove_last_single_frames, self)
    if result is not None and all(v is not None for v in result):
      return sum(result)
    return 0
  def _remove_last_single_frames(self, page):
    return page.remove_last_single_frames()
  def remove_last_single_frames_2(self, threshold, default_frame):
    for page in self:
      page.threshold = threshold
      page.default_frame = default_frame
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._remove_last_single_frames_2, self)
    if result is not None and all(v is not None for v in result):
      return sum(result)
    return 0
  def remove_dark_border(self, threshold, limit, valid):
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.valid=valid
    if MULTIPROCESSING:
      with Pool(processes=N_PROCESSES) as mp_pool:
        result = mp_pool.map(self._remove_dark_border, self)
      if result is not None and all(v is not None for v in result):
        return sum(result)
      return 0
    else:
      count = 0
      for page in self:
        count += self._remove_dark_border(page)
      return count
  def _remove_dark_border(self, page):
    return page.remove_dark_border()
  def remove_gradient_border(self, threshold, limit, valid):
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.valid=valid
    if MULTIPROCESSING:
      with Pool(processes=N_PROCESSES) as mp_pool:
        result = mp_pool.map(self._remove_gradient_border, self)
      if result is not None and all(v is not None for v in result):
        return sum(result)
      return 0
    else:
      count = 0
      for page in self:
        count += self._remove_gradient_border(page)
      return count
  def _remove_gradient_border(self, page):
    return page.remove_gradient_border()
  def remove_fix_border(self, check, limit, max, color, border):
    for page in self:
      page.limit = limit
      page.check = check
      page.max = max
      page.color = color
      page.border = border
    if MULTIPROCESSING:
      with Pool(processes=N_PROCESSES) as mp_pool:
        result = mp_pool.map(self._remove_fix_border, self)
      if result is not None and all(v is not None for v in result):
        return sum(result)
      return 0
    else:
      count = 0
      for page in self:
        count += self._remove_dark_border(page)
      return count
  def _remove_fix_border(self, page):
    return page.remove_fix_border()
  # def remove_last_single_frames_2(self, threshold, default_frame):
  #   result = 0
  #   for i, page in enumerate(self):
  #     if not i:
  #       ai = page.ais.get_model(ISFIRSTPAGE)
  #       model = ai.model if ai is not None else None
  #     page.threshold = threshold
  #     page.default_frame = default_frame
  #     page.model = model
  #   if model is None:
  #     with Pool(processes=N_PROCESSES) as mp_pool:
  #       result = mp_pool.map(self._remove_last_single_frames_2, self)
  #     if result is not None and all(v is not None for v in result):
  #       return sum(result)
  #     return 0
  #   else:
  #     for page in self:
  #       result += self._remove_last_single_frames_2(page)
  #     return result
  def _remove_last_single_frames_2(self, page):
    return page.remove_last_single_frames_2()
  def delete_gray_on_borders(self, threshold, default_frame, color):
    for page in self:
      page.threshold = threshold
      page.default_frame = default_frame
      page.color = color
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._delete_gray_on_borders, self)
    if result is not None and all(v is not None for v in result):
      return sum(result)
    return 0
  def _delete_gray_on_borders(self, page):
    return page.delete_gray_on_borders()
  def center_block(self, threshold, color, use_ai, only_x):
    for page in self:
      page.threshold = threshold
      page.color = color
      page.use_ai = use_ai
      page.only_x = only_x
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._center_block, self)
      if result is not None and all(v is not None for v in result):
        count = sum(result)
    return count
  def _center_block(self, page):
    return page.center_block()
  def set_image_file_name(self):
    for page in self:
      page.set_file_names()
      if page.original_file_name != page.file_name:
        new_file_name = os.path.join(page.original_path, page.file_name + pathlib.Path(page.original_image).suffix)
        if Path(page.original_image).is_file():
          image = Image.open(page.original_image)
          w, h = image.size
          if h > w:
            os.rename(page.original_image, new_file_name)
            page.file_name = new_file_name
            page.newspaper.file_path = new_file_name
  def force_rename_image_file_name(self, n_page):
    for page in self:
      page.newspaper.n_page = n_page
      page.set_file_names()
      if page.original_file_name != page.file_name:
        new_file_name = os.path.join(page.original_path, page.file_name + pathlib.Path(page.original_image).suffix)
        if Path(page.original_image).is_file():
          os.rename(page.original_image, new_file_name)
          page.file_name = new_file_name
          page.newspaper.file_path = new_file_name
      n_page += 1
  def set_bobine_image_file_name(self, name, period):
    n_page = 1
    for page in self:
      page.newspaper.n_page = 1
      new_file = ('00000' + str(n_page))[-4:] + ' - ' + name + ' - ' + period + '.tif'
      if page.original_file_name != new_file:
        if Path(page.original_image).is_file():
          os.rename(os.path.join(page.original_path, page.original_image), os.path.join(page.original_path, new_file))
          page.file_name = new_file
          page.newspaper.file_path = new_file
      n_page += 1
  def convert_images(self, converts):
    if converts is None:
      return
    for page in self:
      page.add_conversion(converts)
    if len(self):
      start_time = time.time()
      if self.date is not None:
        print(f'Starting converting images of of \'{self.newspaper_name}\' ({self.new_root}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      else:
        print(f'Starting converting images of of \'{self.newspaper_name}\' ({self.new_root}) at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      if MULTIPROCESSING:
        with Pool(processes=N_PROCESSES) as mp_pool:
          mp_pool.map(self.convert_image, self)
      else:
        for page in self:
          self.convert_image(page)
      print(f'Conversion of {len(self)} images of \'{self.newspaper_name}\' ({self.new_root}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
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
    n_unkown = 0
    while not end_flag:
      for page in self:
        if next_page > 1:
          next_page -= 1
          continue
        file_to_be_changing, end_flag, next_page, n_unkown = page.rename_pages_files(file_to_be_changing, n_unkown, model)
        if end_flag:
          break
        if next_page >= 0:
          break
      # if next_page < 0:
      #   end_flag = True
    flag = False
    filedirs = []
    numbers = []
    max = 0
    for i, (old_file, new_file) in enumerate(file_to_be_changing):
      n = new_file.split('/')[-1].split('p')[1]
      if n[0] != '?' and n.isdigit():
        numbers.append(int(n))
        max = max(max, int(n))
    missing = [(e1 + 1) for e1, e2 in zip(numbers, numbers[1:]) if e2 - e1 != 1]
    for i in range(max + 1, len(file_to_be_changing) + max):
      missing.append(i)
    _file_to_be_changing = []
    j = 0
    for i, (old_file, new_file) in enumerate(file_to_be_changing):
      n = new_file.split('/')[-1].split('p')[1]
      if n[0] == '?':
        on = new_file.split('_')[-1][1:]
        new_file = '/'.join(new_file.split('/')[:-1]) + '_p' + ('00000' + str(missing[j]))[-len(on):]
        j += 1
      _file_to_be_changing.append((old_file, new_file))
    file_to_be_changing = _file_to_be_changing
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
          # page.add_jpg_metadata()
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
      if self.date is not None:
        print(f'{self.newspaper_name} del giorno {str(self.date.strftime("%d/%m/%Y"))} di tipo \'{type}\' non ha il pdf.')
      else:
        print(f'{self.newspaper_name} di tipo \'{type}\' non ha il pdf.')
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
          if self.date is not None:
            print(f'{self.newspaper_name} del giorno {str(self.date.strftime("%d/%m/%Y"))} di tipo \'{type}\' non ha il jpg di tipo {convert.image_path} con dpi={convert.dpi}')
          else:
            print(
              f'{self.newspaper_name} di tipo \'{type}\' non ha il jpg di tipo {convert.image_path} con dpi={convert.dpi}')
          if integrate:
            self.convert_images([convert])
  def set_bobine_merge_images(self):
    groups_files = []
    file1 = None
    file2 = None
    files = []
    for page in self:
      if Path(page.original_image).stem[:5] != 'merge':
        files.append(page.original_image)
    files.sort()
    for file in files:
      if file1 is None:
        file1 = file
        continue
      if file2 is None:
        file2 = file
        continue
      groups_files.append((file1, file2, file))
      file1 = file2
      file2 = file
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._set_bobine_merge_images, groups_files)
    for file in files:
      os.remove(file)
    return len(files)
  def _set_bobine_merge_images(self, couple_files):
    img1 = cv2.imread(couple_files[0], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(couple_files[1], cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(couple_files[2], cv2.IMREAD_GRAYSCALE)
    vis = np.concatenate((img1, img2, img3), axis=1)
    n = (couple_files[0].split('_')[-1]).split('.')[0]
    file = os.path.join(self.filedir, 'merge_' + n + '.tif')
    cv2.imwrite(file, vis)
    return 1
  def set_bobine_select_images(self, remove_merge, debug, threshold=None):
    for page in self:
      page.remove_merge = remove_merge
      page.debug = debug
      page.threshold = threshold
      page.filedir = self.filedir
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._set_bobine_select_images, self)
    try:
      count = 0
      for elements in result:
        if elements is None:
          continue
        count += elements[1]
      return count
    except:
      return 0
  def _set_bobine_select_images(self, page):
    try:
      return page.set_bobine_select_images()
    except:
      return 0

  def _page_sort(self, page):
    return page.original_image
  def bobine_delete_copies(self):
    _file = None
    file3 = None
    _hash = None
    hash3 = None
    _img = None
    img3 = None
    count = 0
    _ow = None
    _oh = None
    pages = []
    for page in self:
      pages.append(page)
    pages.sort(key=self._page_sort)
    for page in pages:
      file = page.original_image
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      # img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
      if img is None:
        pass
      oh, ow = img.shape
      # img = img[1000:h-2000,1000:w-2000]
      img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)
      hash = imagehash.average_hash(Image.fromarray(img))
      if img3 is not None:
        score, _ = structural_similarity(img, img3, full=True)
        if DEBUG:
          print(score, abs(hash - hash3), os.path.basename(file3), os.path.basename(file))
        if score > SCORECUTOFF or abs(hash - hash3) <= HASHCUTOFF: # or (ow == ow3 and oh == oh3):
          if not DEBUG:
            try:
              n1 = int(file3.split('_')[-2])
              n2 = int(file.split('_')[-2])
              if n2 - n1 <= 2:
                os.remove(file3)
            except:
              continue
          count += 1
      if ow > oh:
        file3 = file
        hash3 = hash
        img3 = img
        ow3 = ow
        oh3 = oh
      else:
        file3 = _file
        hash3 = _hash
        img3 = _img
        ow3 = _ow
        oh3 = _oh
      _file = file
      _hash = hash
      _img = img
      _ow = ow
      _oh = oh
    return count
  def rotate_frames(self, limit=4000, threshold=200, angle=None):
    count = 0
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.angle = angle
    with Pool(processes=N_PROCESSES) as mp_pool:
      counts = mp_pool.map(self._rotate_frames, self)
    for i in counts:
      count += i
    return count
  def _rotate_frames(self, page):
    return page.rotate_frames()
  def rotate_final_frames(self, limit=4000, threshold=200, angle=None):
    count = 0
    for page in self:
      page.limit = limit
      page.threshold = threshold
      page.angle = angle
    with Pool(processes=N_PROCESSES) as mp_pool:
      counts = mp_pool.map(self._rotate_final_frames, self)
    for i in counts:
      count += i
    return count
  def _rotate_final_frames(self, page):
    return page.rotate_final_frames()
  def set_fotogrammi_folders(self, model_path):
    # self.set_GPUs()
    if not os.path.join(STORAGE_BASE, model_path):
      print(f'{model_path} doesn\'t exist.')
      return
    model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, model_path))
    count = 0
    for page in self:
      if page.newspaper.is_first_page(model):
        print(page.original_image)
        count += 1
    return count
  def _set_fotogrammi_folders(self, page):
    try:
      return page.set_fotogrammi_folders()
    except:
      return 0
  def reset_dpi(self, ):
    with Pool(processes=N_PROCESSES) as mp_pool:
      result = mp_pool.map(self._reset_dpi, self)
    return sum(result)
  def _reset_dpi(self, page):
    return page.reset_dpi()



