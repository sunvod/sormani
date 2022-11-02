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
import cv2

from pathlib import Path
from PIL import Image
from pathlib import Path
from argparse import Namespace
from contextlib import suppress
from multiprocessing import Pool
from os import listdir
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

ORIGINAL_DPI = 400
UPSAMPLING_DPI = 700

log = logging.getLogger('ocrmypdf')

class Page:
  def __init__(self, file_name, newspaper_name, original_image, pdf_path, jpg_path, txt_path):
    self.file_name = file_name
    self.newspaper_name = newspaper_name
    self.original_image = original_image
    self.pdf_path = pdf_path
    self.jpg_path = jpg_path
    self.txt_path = txt_path
    self.conversions = []

  def add_conversion(self, conversion):
    if isinstance(conversion, list):
      for conv in conversion:
        self.conversions.append(conv)
    else:
      self.conversions.append(conversion)
class Conversion:
  def __init__(self, image_path, dpi, quality, resolution):
    self.image_path = image_path
    self.dpi = dpi
    self.quality = quality
    self.resolution = resolution


def sigbus(*args):
  raise InputFileError("Lost access to the input file")

def get_files(newspaper_name, root, ext, tiff_path ='Tiff_images', path_exclude = [], path_exist ='pdf', force = False):
  file_pool = []
  for filedir, dirs, files in os.walk(os.path.join(root, tiff_path, newspaper_name)):
    if filedir in path_exclude:
      continue
    for file in files:
      dir_in_filedir = filedir.split('/')
      dir_in_filedir = list(map(lambda x: x.replace(tiff_path, 'Jpg-Pdf'), dir_in_filedir))
      if pathlib.Path(file).suffix == '.' + ext:
        _file = file[: len(file) - len(ext) - 1]
        files_existing = None
        if os.path.isdir(os.path.join('/'.join(dir_in_filedir), path_exist)):
          files_existing = [f for f in listdir(os.path.join('/'.join(dir_in_filedir), path_exist))]
        if force or files_existing is None or not _file + '.pdf' in files_existing:
          file_name = Path(file).stem
          page = Page(file_name, newspaper_name, os.path.join(filedir, file), '/'.join(dir_in_filedir), '/'.join(dir_in_filedir), root)
          file_pool.append(page)
  return file_pool

def convert_image(page):
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
        im.save(file, 'JPEG', dpi=(convert.dpi, convert.dpi), quality = convert.quality)
  except Exception:
    #tb = sys.exc_info()
    pass
def convert_images(page_pool, converts):
  if converts is None:
    return
  for page in page_pool:
    page.add_conversion(converts)
  start_time = time.time()
  print(f'Starting converting images of \'{page.newspaper_name}\' at {str(datetime.datetime.now())}')
  if len(page_pool):
    with Pool(processes=14) as mp_pool:
      mp_pool.map(convert_image, page_pool)
    print(f'Conversion of {len(page_pool)} images ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
  else:
    print(f'Warning: There is no files to convert.')


def create_files(name, root, ext = 'tif', converts = [Conversion('jpg_small', 150, 90, 2000), Conversion('jpg_medium', 300, 90, 2000)]):
  start_time = time.time()
  page_pool = get_files(name, root, ext)
  if len(page_pool):
    print(f'Starting creating pdf/a of \'{name}\' at {str(datetime.datetime.now())}')
    with Pool(processes=14) as mp_pool:
      mp_pool.map(to_pdfa, page_pool)
      #mp_pool.map(exec_ocrmypdf, file_pool)
    print(f'Creation of {len(page_pool)} pdf/a files for \'{name}\' ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
  else:
    print(f'Warning: There is no files to process for \'{name}\'.')
  if converts is not None:
    page_pool = get_files(name, root, ext, force=True)
    convert_images(page_pool, converts)


def totext(filename):
    doc = fitz.open(filename)
    toc = doc.get_toc()
    page = doc[0]
    pix = page.get_pixmap()
    print(pix.height) # 4900x7088
    print(pix.width)
    text = page.get_text('text')
    print(text)


def exec_ocrmypdf(dirs):
  name = dirs[0]
  file_name = dirs[1]
  pdf_base = dirs[2]
  root = dirs[3]
  Path(os.path.join(pdf_base, 'pdf')).mkdir(parents=True, exist_ok=True)
  Path(os.path.join(root, 'txt', name)).mkdir(parents=True, exist_ok=True)
  ocrmypdf.configure_logging(verbosity = -1)
  ocrmypdf.ocr(
    file_name + '.tif',
    pdf_base + '/pdf/' + os.path.basename(os.path.realpath(file_name)) + '.pdf',
    image_dpi = ORIGINAL_DPI,
    oversample = UPSAMPLING_DPI,
    output_type = 'pdfa')

def to_pdfa(page):
  _parser, options, plugin_manager = get_parser_options_plugins(None)
  options = Namespace()
  Path(os.path.join(page.pdf_path, 'pdf')).mkdir(parents=True, exist_ok=True)
  Path(os.path.join(page.txt_path, 'txt', page.newspaper_name)).mkdir(parents=True, exist_ok=True)
  options.input_file = page.original_image
  options.output_file = os.path.join(page.pdf_path, 'pdf', page.file_name) + '.pdf'
  options.sidecar = os.path.join(page.txt_path, 'txt', page.newspaper_name, page.file_name) + '.txt'
  options.author = None
  options.clean = False
  options.clean_final = False
  options.deskew = False
  options.fast_web_view = 1.0
  options.force_ocr = False
  options.image_dpi = ORIGINAL_DPI
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
  options.oversample = UPSAMPLING_DPI
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
  create_files('La Stampa', '/mnt/storage01/sormani')


