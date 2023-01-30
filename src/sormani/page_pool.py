from __future__ import annotations

import multiprocessing
import os
import pathlib
import time
import datetime
import cv2
import numpy as np

from multiprocessing import Pool
from src.sormani.system import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# global_count = multiprocessing.Value('I', 0)
global_count_contrast = multiprocessing.Value('I', 0)

class Page_pool(list):
  def __init__(self, newspaper_name, filedir, name_complete, new_root, date, force = False):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.new_root = new_root
    self.name_complete = name_complete
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
      print(f'{self.newspaper_name} ({self.name_complete}) del giorno {str(self.date.strftime("%d/%m/%y"))} ha le pagine esatte (code: {countminusone} {countzero} {countplusone}).')
    else:
      msg = '{} ({}) del giorno {} ha le pagine {} non esatte  (code: {} {} {}).'.format(self.newspaper_name, self.name_complete, str(self.date.strftime("%d/%m/%y")), errors, countminusone, countzero, countplusone)
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
      print(f'Start creating pdf/a of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.to_pdfa, self)
      print(f'The creation of {len(self)} pdf/a files for of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
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
  def change_contrast(self, contrast = 50, force = True, number = None, ocr = True):
    if len(self):
      self.number = number
      self.ocr = ocr
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      print(f'Start changing the contrast of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
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
  def change_threshold(self, force = True, limit = 50, color = 255, inversion = False):
    if len(self):
      start_time = time.time()
      dir_name = self.filedir.split('/')[-1]
      print(f'Start changing the threshold of \'{self.newspaper_name}\' ({dir_name}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      for page in self:
        page.limit = limit
        page.color = color
        page.inversion = inversion
        page.force = force
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self._change_threshold, self)
      print(f'The {len(self)} pages threshold change of \'{self.newspaper_name}\' ({dir_name}) ends at {str(datetime.datetime.now().strftime("%H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to changing the threshold for \'{self.newspaper_name}\'.')
  def _change_threshold(self, page):
    global global_count_contrast
    i = page.change_threshold()
    with global_count_contrast.get_lock():
      global_count_contrast.value += i
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
      file_name_no_ext = Path(page.original_image).stem
      file_path_no_ext = os.path.join(self.filedir, file_name_no_ext)
      ext = Path(page.original_image).suffix
      image = Image.open(page.original_image)
      width, height = image.size
      if width < height:
        if flag:
          os.rename(page.original_image, file_path_no_ext + '_0' + ext)
        continue
      pages.append(page)
    # with Pool(processes=N_PROCESSES) as mp_pool:
    #   mp_pool.map(self._divide_image, pages)
    for page in pages:
      self._divide_image(page)
  def _divide_image(self, page):
    i = 0
    try:
      file_name_no_ext = Path(page.original_image).stem
      file_path_no_ext = os.path.join(self.filedir, file_name_no_ext)
      ext = Path(page.original_image).suffix
      img = cv2.imread(page.original_image, cv2.IMREAD_GRAYSCALE)
      image1, image2 = page.newspaper.divide(img)
      image1.save(file_path_no_ext + '_1' + ext)
      image2.save(file_path_no_ext + '_2' + ext)
      os.remove(page.original_image)
      i += 1
    except Exception as e:
      with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
        sormani_log.write('No valid Image: ' + page.original_image + '\n')
      print(f'Not a valid image: {page.original_image}')
    if i:
      print('.', end='')
      with global_count.get_lock():
        global_count.value += i
        if global_count.value % 100 == 0:
          print()
  def remove_borders(self):
    with Pool(processes=N_PROCESSES) as mp_pool:
      mp_pool.map(self._remove_borders, self)
  def _remove_borders(self, page):
    i = 0
    try:
      file_name_no_ext = Path(page.original_image).stem
      image = Image.open(page.original_image)
      width, height = image.size
      lr = 0 if file_name_no_ext.split('_')[-1] == '1' else 1
      parameters = page.newspaper.get_remove_borders_parameters(lr, width, height)
      img = image.crop((parameters.left, parameters.top, parameters.right, parameters.bottom))
      img.save(page.original_image)
      i += 1
    except Exception as e:
      with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
        sormani_log.write('No valid Image: ' + page.original_image + '\n')
      print(f'Not a valid image: {page.original_image}')
    if i:
      print('.', end='')
      with global_count.get_lock():
        global_count.value += i
        if global_count.value % 100 == 0:
          print()
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
      print(f'Starting converting images of of \'{self.newspaper_name}\' ({self.new_root}) of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%H:%M:%S"))}')
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.convert_image, self)
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
  def set_bobine_images(self):
    couple_files = []
    file1 = None
    files = []
    for page in self:
      if Path(page.original_image).stem[:5] != 'merge':
        files.append(page.original_image)
    files.sort()
    for file in files:
      if file1 is None:
        file1 = file
        continue
      couple_files.append((file1, file))
      file1 = file
    i = 1
    file2 = None
    for file1, file2 in couple_files:
      img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
      img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
      vis = np.concatenate((img1, img2), axis=1)
      n = '00' + str(i) if i < 10 else '0' + str(i) if i < 100 else str(i)
      file3 = os.path.join(self.filedir, 'merge_' + n + '.tif')
      cv2.imwrite(file3, vis)
      i += 1
      os.remove(file1)
    if file2 is not None:
      os.remove(file2)

  def set_bobine_merges(self):
    def _order(e):
      return e[0]
    files = []
    for page in self:
      if Path(page.original_image).stem[:5] == 'merge':
        files.append(page.original_image)
    files.sort()
    j = 1
    for file in files:
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      ret, thresh = cv2.threshold(img, 127, 255, 0)
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      bimg = img.copy()
      bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
      books = []
      for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 4000 and h > 4000:
          books.append((x, y, w, h))
      books.sort(key=_order)
      if len(books) > 1:
        couple_books = []
        book1 = None
        for i, book in enumerate(books):
          if book1 is None:
            book1 = book
            continue
          couple_books.append((book1, book))
          book1 = book
          if i == len(books) - 1:
            couple_books.append((book, None))
            break
        books = []
        bw = 0
        for book1, book2 in couple_books:
          if book2 is not None and abs(book1[0] + book1[2] - book2[0]) < 100:
            _x = book1[0]
            _y = book1[1]
            _w = book1[2] + book2[2] + book2[0] - (book1[0] + book1[2])
            _h = book1[3] + book2[3] + book2[1] - (book1[1] + book1[3])
            books.append((_x, _y, _w, _h))
            if _w > bw:
              bw = _w
          else:
            books.append(book1)
            if book1[2] > bw:
              bw = book1[2]
        for x, y, w, h in books:
          if w == bw:
            cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 5)
            break
      elif len(books) == 1:
        book = books[0]
        x = book[0]
        y = book[1]
        w = book[2]
        h = book[3]
        cv2.rectangle(bimg, (x, y), (x + w, y + h), (0, 255, 0), 5)
      else:
        continue
      n = '00' + str(j) if j < 10 else '0' + str(j) if j < 100 else str(j)
      file2 = os.path.join(self.filedir, 'fotogrammi_' + n + '.tif')
      _x, _y, _w, _h = cv2.boundingRect(img)
      if x != 0 and x + w != _w:
        roi = img[y:y + h, x:x + w]
        cv2.imwrite(file2, roi)
      j += 1
    for file in files:
      os.remove(file)
  def rotate_fotogrammi(self, verbose = False, limit=4000):
    def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    count = 1
    for page in self:
      file = page.original_image
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      ret, thresh = cv2.threshold(img, 127, 255, 0)
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      bimg = img.copy()
      bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
      books = []
      rect = None
      for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > limit and h > limit:
          books.append(contour)
      for contour in books:
        rect = cv2.minAreaRect(contour)
        if verbose:
          box = np.int0(cv2.boxPoints(rect))
          cv2.drawContours(bimg, [box], 0, (0, 255, 0), 3)
      if rect is not None:
        angle = rect[2]
        if angle < 45:
          angle = 90 - angle
        if angle > 85:
          bimg = rotate_image(bimg, angle - 90)
          cv2.imwrite(file, bimg)
          count += 1
        elif verbose:
          cv2.imwrite(file, bimg)

  def rotate_page(self, verbose=False, limit=1000):
    def _rotate_page(e):
      return e.original_file_name
    def rotate_image(image, angle):
      image_center = tuple(np.array(image.shape[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
      result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
      return result
    count = 1
    self.sort(key=_rotate_page)
    for page in self:
      file = page.original_image
      img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      _, thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
      # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
      # ext = Path(file).suffix
      # file = '.'.join(file.split('.')[:-1]) + '_' + str(1) + ext
      # dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
      # dilation = 255 - dilation
      # ext = Path(file).suffix
      # new_file = '.'.join(file.split('.')[:-1]) + '_' + '_' + ext
      # cv2.imwrite(new_file, dilation)
      contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      bimg = img.copy()
      bimg = cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB)
      books = []
      rect = None
      for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > limit and h > limit:
          books.append(contour)
      angle = None
      for contour in books:
        rect = cv2.minAreaRect(contour)
        if rect[2] > 40:
          angle = rect[2]
          if verbose:
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(bimg, [box], 0, (0, 255, 0), 10)
      if rect is not None and angle is not None:
        if angle < 45:
          angle = 90 - angle
        bimg = rotate_image(bimg, angle - 90)
        cv2.imwrite(file, bimg)
        # ext = Path(file).suffix
        # new_file = '.'.join(file.split('.')[:-1]) + '_' + str(angle) + ext
        # os.rename(file, new_file)
        count += 1