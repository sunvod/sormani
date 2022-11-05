#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""ocrmypdf command line entrypoint."""

from __future__ import annotations

import logging
import os
import signal
import sys
import pathlib
import fitz
import time
import datetime
import multiprocessing
import ocrmypdf
import datetime
import cv2
import pprint

from pathlib import Path
from PIL import Image
from pathlib import Path
from argparse import Namespace
from contextlib import suppress
from multiprocessing import Pool
from os import listdir
from PyPDF2 import PdfFileReader, PdfFileMerger
from os.path import isfile, join
from ocrmypdf import __version__
from ocrmypdf._plugin_manager import get_parser_options_plugins
from ocrmypdf._sync import run_pipeline
from ocrmypdf._validation import check_options
from ocrmypdf.api import Verbosity, configure_logging
from ocrmypdf.exceptions import (
  BadArgsError,
  ExitCode,
  InputFileError,
  MissingDependencyError,
)

N_PROCESSES = 14
ORIGINAL_DPI = 400
UPSAMPLING_DPI = 400
MONTHS = ['GEN', 'FEB', 'MAR', 'APR', 'MAG', 'GIU', 'LUG', 'AGO', 'SET', 'OTT', 'NOV', 'DIC']

log = logging.getLogger('ocrmypdf')

class Page:
  def __init__(self, file_name, date, newspaper_name, original_image, pdf_path, jpg_path, txt_path):
    self.original_file_name = file_name
    self.file_name = file_name
    self.original_image = original_image
    self.original_path = str(Path(original_image).parent.resolve())
    dir_in_filedir = self.original_path.split('/')
    self.date = date
    self.year = date.year
    self.month = date.month
    self.month_text = MONTHS[self.month - 1]
    self.day = date.day
    self.newspaper = Newspaper.create(original_image, newspaper_name, date)
    self.pdf_path = pdf_path
    self.pdf_file_name = os.path.join(self.pdf_path, 'pdf', self.file_name) + '.pdf'
    self.jpg_path = jpg_path
    self.txt_path = os.path.join(txt_path, 'txt', self.newspaper.name)
    self.txt_file_name = os.path.join(txt_path, 'txt', self.newspaper.name, self.file_name) + '.txt'
    self.original_txt_file_name = self.txt_file_name
    self.conversions = []
  def add_conversion(self, conversion):
    if isinstance(conversion, list):
      for conv in conversion:
        self.conversions.append(conv)
    else:
      self.conversions.append(conversion)
  def set_file_names(self):
    self.file_name = self.newspaper.name.replace(' ', '_') \
                     + '_' + str(self.year) \
                     + '_' + str(self.month_text) \
                     + '_' + (str(self.day) if self.day >= 10 else '0' + str(self.day)) \
                     + '_p' + (str(self.newspaper.n_page) if self.newspaper.n_page >= 10 else '0' + str(self.newspaper.n_page))
    txt_file_name = self.newspaper.name.replace(' ', '_') \
                     + '_' + str(self.year) \
                     + '_' + (str(self.month) if self.month >= 10 else '0' + str(self.month)) \
                     + '_' + (str(self.day) if self.day >= 10 else '0' + str(self.day)) \
                     + '_p' + (str(self.newspaper.n_page) if self.newspaper.n_page >= 10 else '0' + str(self.newspaper.n_page))
    self.txt_file_name = os.path.join(self.txt_path, txt_file_name) + '.txt'

class Conversion:
  def __init__(self, image_path, dpi, quality, resolution):
    self.image_path = image_path
    self.dpi = dpi
    self.quality = quality
    self.resolution = resolution

class Page_pool(list):
  def __init__(self, newspaper_name, date):
    self.newspaper_name = newspaper_name
    self.date = date
  def set_pages(self):
    self.sort(key = self._set_pages_sort)
    n_pages = len(self)
    for n_page, page in enumerate(self):
      page.newspaper.n_pages = n_pages
      page.newspaper.set_n_page(n_page, self.date)
    self.sort(key=self._n_page_sort)
  def _set_pages_sort(self, page):
    return page.original_file_name
    #return os.path.getmtime(Path(page.original_image))
  def _n_page_sort(self, page):
    return page.newspaper.n_page
  def create_pdf(self):
    if len(self):
      start_time = time.time()
      print(f'Starting creating pdf/a of \'{self.newspaper_name}\' at {str(datetime.datetime.now())}')
      self.set_pages()
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.to_pdfa, self)
        # mp_pool.map(exec_ocrmypdf, file_pool)
      print(f'Creation of {len(self)} pdf/a files for \'{self.newspaper_name}\' ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to process for \'{self.newspaper_name}\'.')
  def to_pdfa(self, page):
    _parser, options, plugin_manager = get_parser_options_plugins(None)
    options = Namespace()
    Path(os.path.join(page.pdf_path, 'pdf')).mkdir(parents=True, exist_ok=True)
    Path(page.txt_path).mkdir(parents=True, exist_ok=True)
    exec_ocrmypdf(page.original_image, page.pdf_file_name, page.txt_file_name, ORIGINAL_DPI, UPSAMPLING_DPI)

    os.rename(page.pdf_file_name, page.pdf_file_name + '.2')
    file_in = open(page.pdf_file_name + '.2', 'rb')
    pdf_reader = PdfFileReader(file_in)
    # metadata = pdf_reader.getDocumentInfo()
    # pprint.pprint(metadata)
    pdf_merger = PdfFileMerger()
    pdf_merger.append(file_in)
    pdf_merger.addMetadata({
      '/Keywords': 'Nome del periodico:' + page.newspaper.name
                   + ' ; Anno:' + str(page.year)
                   + ' ; Mese:' + str(page.month)
                   + ' ; Giorno:' + str(page.day)
                   + ' ; Numero del quotidiano:' + page.newspaper.number
                   + ' ; Anno del quotidiano:' + page.newspaper.year
    })
    file_out = open(page.pdf_file_name, 'wb')
    pdf_merger.write(file_out)
    file_in.close()
    file_out.close()
    os.remove(page.pdf_file_name + '.2')

    # os.rename(page.pdf_file_name, page.pdf_file_name + '.2')
    # reader = PdfFileReader(page.pdf_file_name + '.2')
    # writer = PdfFileWriter()
    # writer.appendPagesFromReader(reader)
    # metadata = reader.getDocumentInfo()
    # writer.addMetadata(metadata)
    # writer.addMetadata({"/Nome del periodico": page.newspaper.name})
    # writer.addMetadata({"/Nome del periodico": page.newspaper.name, '/Anno': page.year, '/Mese': page.month, '/Giorno': page.day})
    # if page.newspaper.number is not None:
    #   writer.addMetadata({'/Numero del quotidiano': page.newspaper.number})
    # if page.newspaper.year is not None:
    #   writer.addMetadata({'/Numero del quotidiano': page.newspaper.year})
    # with open(page.pdf_file_name, "wb") as fp:
    #   writer.write(fp)
  def set_files_name(self):
    for page in self:
      page.set_file_names()
      if page.original_file_name != page.file_name:
        new_file_name = page.txt_file_name.replace(page.original_file_name, page.txt_file_name)
        if Path(page.original_txt_file_name).is_file():
          os.rename(page.original_txt_file_name, new_file_name)
          page.txt_path = new_file_name
        new_file_name = page.pdf_file_name.replace(page.original_file_name, page.file_name)
        if Path(page.pdf_file_name).is_file():
          os.rename(page.pdf_file_name, new_file_name)
          page.pdf_file_name = new_file_name
        new_file_name = page.original_image.replace(page.original_file_name, page.file_name)
        if Path(page.original_image).is_file():
          os.rename(page.original_image, new_file_name)
          page.original_image = new_file_name
  def convert_image(self, page):
    try:
      im = Image.open(page.original_image)
      for convert in page.conversions:
        path_image = os.path.join(page.jpg_path, convert.image_path)
        Path(path_image).mkdir(parents=True, exist_ok=True)
        file = os.path.join(path_image, page.file_name) + '.jpg'
        if not Path(file).is_file():
          if im.size[0] < im.size[1]:
            wpercent = (convert.resolution / float(im.size[1]))
            xsize = int((float(im.size[0]) * float(wpercent)))
            im = im.resize((xsize, convert.resolution), Image.Resampling.LANCZOS)
          else:
            wpercent = (convert.resolution / float(im.size[0]))
            ysize = int((float(im.size[1]) * float(wpercent)))
            im = im.resize((convert.resolution, ysize), Image.Resampling.LANCZOS)
          im.save(file, 'JPEG', dpi=(convert.dpi, convert.dpi), quality=convert.quality)
    except Exception:
      # tb = sys.exc_info()
      pass
  def convert_images(self, converts):
    if converts is None:
      return
    for page in self:
      page.add_conversion(converts)
    if len(self):
      start_time = time.time()
      print(f'Starting converting images of \'{self.newspaper_name}\' at {str(datetime.datetime.now())}')
      with Pool(processes=14) as mp_pool:
        mp_pool.map(self.convert_image, self)
      print(
        f'Conversion of {len(self)} images ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to convert for \'{self.newspaper_name}\'.')


class Newspaper():
  @staticmethod
  def create(file_name, name, date):
    if name == 'La Stampa':
      newspaper = La_stampa(file_name, date)
    elif name == 'Il Manifesto':
      newspaper = Il_manifesto(file_name, date)
    if name == 'Avvenire':
      newspaper = Avvenire(file_name, date)
    return newspaper
  def __init__(self, name, file_name, date):
    self.name = name
    self.file_name = file_name
    self.date = date
    self.year, self.number = self.get_head()
  def set_n_page(self, n_page, date):
    year = self.file_name[-15:-11]
    month = self.file_name[-10:-7]
    day = self.file_name[-6:-4]
    if month in MONTHS:
      month = str(MONTHS.index(self.file_name[-10:-7]) + 1)
    if year.isdigit() and month.isdigit() and day.isdigit():
      file_date = datetime.date(int(year), int(month), int(day))
      return date == file_date
    return False
  def crop(self, left, top, right, bottom):
    image = Image.open(self.file_name)
    image = image.crop((left, top, right, bottom))
    image.save('temp.tif')
    exec_ocrmypdf('temp.tif', oversample=800)
    f = open("temp.txt", "r")
    x = f.read()
    print(x)
    os.remove('temp.tif')
    os.remove('temp.pdf')
    os.remove('temp.txt')
    return x[:-1]

class La_stampa(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'La Stampa', file_name, date)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    r = n_page % 4
    n = n_page // 4
    if r == 0:
      self.n_page = n * 2 + 1
    elif r == 1:
      self.n_page = self.n_pages - n * 2
    elif r == 2:
      self.n_page = self.n_pages - n * 2 - 1
    else:
      self.n_page = n * 2 + 2
  def get_head(self):
    # text = super().crop(left = 920, top = 1550, right = 1260, bottom = 1650)
    # year = ''.join(filter(str.isdigit, text[4:9]))
    # number = ''.join(filter(str.isdigit, text[-4:]))
    year = str(150 + self.date.year - 2016)
    number = str((self.date - datetime.date(self.date.year, 1, 1)).days)
    return year, number

class Il_manifesto(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'Il Manifesto', file_name, date)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    self.n_page = n_page + 1
class Avvenire(Newspaper):
  def __init__(self, file_name, date):
    Newspaper.__init__(self, 'Avvenire', file_name, date)
  def set_n_page(self, n_page, date):
    if super().set_n_page(n_page, date):
      self.n_page = n_page + 1
      return
    self.n_page = n_page + 1

# class Images_group():
#   def __init__(self, root, tiff_path, newspaper_name, path_exclude):
#     self.elements = []
#     for filedir, dirs, files in os.walk(os.path.join(root, tiff_path, newspaper_name)):
#       n_pages = len(files)
#       if filedir in path_exclude or n_pages == 0:
#         continue
#       self.elements.append((filedir, files))
#     self.i = 0
#   def __iter__(self):
#     return self
#   def __next__(self):
#     x = self.elements[self.i]
#     self.i += 1
#     return x

class Images_group():

  def __init__(self, filedir, files):
    self.filedir = filedir
    self.files = files
    year = ''.join(filter(str.isdigit, filedir.split('/')[-3]))
    month = ''.join(filter(str.isdigit, filedir.split('/')[-2]))
    day = ''.join(filter(str.isdigit, filedir.split('/')[-1]))
    if year.isdigit() and month.isdigit() and day.isdigit():
      self.date = datetime.date(int(year), int(month), int(day))
    else:
      self.date = datetime.today()
  def get_page_pool(self, newspaper_name, root, ext, image_path, path_exist, force):
    page_pool = Page_pool(newspaper_name, self.date)
    for n_page, file in enumerate(self.files):
      dir_in_filedir = self.filedir.split('/')
      dir_in_filedir = list(map(lambda x: x.replace(image_path, 'Jpg-Pdf'), dir_in_filedir))
      if pathlib.Path(file).suffix == '.' + ext:
        _file = file[: len(file) - len(ext) - 1]
        files_existing = None
        if os.path.isdir(os.path.join('/'.join(dir_in_filedir), path_exist)):
          files_existing = [f for f in listdir(os.path.join('/'.join(dir_in_filedir), path_exist))]
        if force or files_existing is None or not _file + '.pdf' in files_existing:
          file_name = Path(file).stem
          page = Page(file_name, self.date, newspaper_name, os.path.join(self.filedir, file), '/'.join(dir_in_filedir), '/'.join(dir_in_filedir), root)
          page_pool.append(page)
    return page_pool
class Sormani():
  def __init__(self,
               newspaper_name,
               root = '/mnt/storage01/sormani',
               ext = 'tif',
               image_path ='Tiff_images',
               path_exclude = [],
               path_exist ='pdf',
               force = False):
    self.newspaper_name = newspaper_name
    self.root = root
    self.ext = ext
    self.image_path = image_path
    self.path_exclude = path_exclude
    self.path_exist = path_exist
    self.force = force
    self.converts = None
    self.elements = []
    for filedir, dirs, files in os.walk(os.path.join(root, image_path, newspaper_name)):
      n_pages = len(files)
      if filedir in path_exclude or n_pages == 0:
        continue
      files.sort()
      self.elements.append(Images_group(filedir, files))
    self.elements.sort(key = self._elements_sort)
    self.i = 0
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
  def create_all_images(self, converts = [Conversion('jpg_small', 150, 60, 2000), Conversion('jpg_medium', 300, 90, 2000)]):
    for page_pool in self:
      page_pool.create_pdf()
      page_pool.set_files_name()
      page_pool.convert_images(converts)

# def exec_ocrmypdf(dirs):
#   name = dirs[0]
#   file_name = dirs[1]
#   pdf_base = dirs[2]
#   root = dirs[3]
#   Path(os.path.join(pdf_base, 'pdf')).mkdir(parents=True, exist_ok=True)
#   Path(os.path.join(root, 'txt', name)).mkdir(parents=True, exist_ok=True)
#   ocrmypdf.configure_logging(verbosity = -1)
#   ocrmypdf.ocr(
#     file_name + '.tif',
#     pdf_base + '/pdf/' + os.path.basename(os.path.realpath(file_name)) + '.pdf',
#     image_dpi = ORIGINAL_DPI,
#     oversample = UPSAMPLING_DPI,
#     output_type = 'pdfa')

def sigbus(*args):
  raise InputFileError("Lost access to the input file")


def exec_ocrmypdf(input_file, output_file = 'temp.pdf', sidecar_file = 'temp.txt', image_dpi = ORIGINAL_DPI, oversample = 0):
  _, _, plugin_manager = get_parser_options_plugins(None)
  options = Namespace()
  options.input_file = input_file
  options.output_file = output_file
  options.sidecar = sidecar_file
  options.image_dpi = image_dpi
  options.oversample = oversample
  options.author = None
  options.clean = False
  options.clean_final = False
  options.deskew = False
  options.fast_web_view = 1.0
  options.force_ocr = False
  options.jbig2_lossy = False
  options.jbig2_page_group_size = 0
  options.jobs = None
  options.jpeg_quality = 0
  options.keep_temporary_files = False
  options.keywords = None
  options.languages = {'ita'}
  options.max_image_mpixels = 1000.0
  options.optimize = 0
  options.output_type = 'pdfa'
  options.pages = None
  options.pdf_renderer = 'auto'
  options.pdfa_image_compression = 'auto'
  options.plugins = []
  options.png_quality = 0
  options.progress_bar = True
  options.quiet = False
  options.redo_ocr = False
  options.remove_background = False
  options.remove_vectors = False
  options.rotate_pages = False
  options.rotate_pages_threshold = 14.0
  options.skip_big = None
  options.skip_text = False
  options.subject = None
  options.tesseract_config = []
  options.tesseract_oem = None
  options.tesseract_pagesegmode = None
  options.tesseract_thresholding = 0
  options.tesseract_timeout = 360.0
  options.title = None
  options.unpaper_args = None
  options.use_threads = True
  options.user_patterns = None
  options.user_words = None
  options.verbose = -1
  with suppress(AttributeError, PermissionError):
    os.nice(5)
  verbosity = options.verbose
  if not os.isatty(sys.stderr.fileno()):
    options.progress_bar = False
  if options.quiet:
    verbosity = Verbosity.quiet
    options.progress_bar = False
  configure_logging(
    verbosity,
    progress_bar_friendly=options.progress_bar,
    manage_root_logger=True,
    plugin_manager=plugin_manager,
  )
  log.debug('ocrmypdf %s', __version__)
  try:
    check_options(options, plugin_manager)
  except ValueError as e:
    log.error(e)
    return ExitCode.bad_args
  except BadArgsError as e:
    log.error(e)
    return e.exit_code
  except MissingDependencyError as e:
    log.error(e)
    return ExitCode.missing_dependency
  with suppress(AttributeError, OSError):
    signal.signal(signal.SIGBUS, sigbus)
  run_pipeline(options=options, plugin_manager=None)

if __name__ == '__main__':
  cpu = multiprocessing.cpu_count()
  Sormani('La Stampa').create_all_images()


