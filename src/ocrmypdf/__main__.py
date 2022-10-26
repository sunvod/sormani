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

log = logging.getLogger('ocrmypdf')


def sigbus(*args):
  raise InputFileError("Lost access to the input file")

def get_files(name, root, ext, path_exclude = [], path_exist = None):
  file_pool = []
  for filedir, dirs, files in os.walk(root + '/' + name):
    if filedir in path_exclude:
      continue
    for file in files:
      if pathlib.Path(file).suffix == '.' + ext:
        _file = file[: len(file) - len(ext) - 1]
        if path_exist is not None:
          files_exist = [f for f in listdir(path_exist)]
        if path_exist is None or not _file + '.pdf' in files_exist:
          file_pool.append([filedir + '/' + _file, root + '/' + name])
  return file_pool

# import cv2, os
# base_path = "data/images/"
# new_path = "data/ims/"
# for infile in os.listdir(base_path):
#     print ("file : " + infile)
#     read = cv2.imread(base_path + infile)
#     outfile = infile.split('.')[0] + '.jpg'
#     cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
def convert_image(args):
  base = args[0]
  name = args[1]
  ext = args[2]
  converts = args[3]
  file_name = os.path.basename(os.path.realpath(name))
  try:
    #im = cv2.imread(name + '.' + ext)
    im = Image.open(name + '.' + ext)
    for c in converts:
      path_image = os.path.join(base, c[2])
      Path(path_image + '/').mkdir(parents=True, exist_ok=True)
      file = os.path.join(path_image, file_name) + '.' + c[1]
      if not Path(file).is_file():
        #cv2.imwrite(file, im, [int(cv2.IMWRITE_JPEG_QUALITY), c[3]])
        #im.save(file, c[0], dpi=(c[3], c[3]))
        im.thumbnail(im.size)
        im.save(file, c[0], dpi=(c[3], c[3]), quality = 90)
  except Exception:
    #tb = sys.exc_info()
    pass
def convert_images(name, file_pool, ext, converts):
  if converts is None:
    return
  image_pool = []
  for file in file_pool:
    image_pool.append([file[1], file[0], ext, converts])
  start_time = time.time()
  print(f'Starting converting images of \'{name}\' at {str(datetime.datetime.now())}')
  if len(image_pool):
    with Pool(processes=14) as mp_pool:
      mp_pool.map(convert_image, image_pool)
    print(f'Conversion of {len(file_pool)} images for \'{name}\' ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
  else:
    print(f'Warning: There is no files to convert for \'{name}\'.')


def create_files(name, root, ext = 'tif', converts = [('jpeg', 'jpg', 'jpg150', 150), ('jpeg', 'jpg', 'jpg300', 300)]):
  path_pdf = root + '/' + name + '/pdf'
  path_txt = root + '/' + name + '/pdf/txt'
  Path(path_pdf + '/').mkdir(parents=True, exist_ok=True)
  Path(path_txt + '/').mkdir(parents=True, exist_ok=True)
  start_time = time.time()
  file_pool = get_files(name, root, ext, path_exclude = [path_pdf, path_txt], path_exist = path_pdf)
  convert_images(name, get_files(name, root, ext, path_exclude = [path_pdf, path_txt]), 'tif', converts)
  if len(file_pool):
    print(f'Starting creating pdf/a of \'{name}\' at {str(datetime.datetime.now())}')
    with Pool(processes=14) as mp_pool:
      mp_pool.map(to_pdfa, file_pool)
    print(f'Creation of {len(file_pool)} pdf/a files for \'{name}\' ends at {str(datetime.datetime.now())} and takes {round(time.time() - start_time)} seconds.')
  else:
    print(f'Warning: There is no files to process for \'{name}\'.')


def totext(filename):
    doc = fitz.open(filename)
    toc = doc.get_toc()
    page = doc[0]
    pix = page.get_pixmap()
    print(pix.height) # 4900x7088
    print(pix.width)
    text = page.get_text('text')
    print(text)

def to_pdfa(dirs):
  _parser, options, plugin_manager = get_parser_options_plugins(None)
  name = dirs[0]
  pdf_base = dirs[1]
  options = Namespace()
  dir_path = os.path.dirname(os.path.realpath(name))
  file_name = os.path.basename(os.path.realpath(name))
  options.input_file = name + '.tif'
  options.output_file = pdf_base + '/pdf/' + file_name + '.pdf'
  options.sidecar = pdf_base + '/pdf/txt/' + file_name + '.txt'
  options.author = None
  options.clean = False
  options.clean_final = False
  options.deskew = False
  options.fast_web_view = 1.0
  options.force_ocr = False
  options.image_dpi = 400
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
  options.oversample = 0
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
  options.tesseract_timeout = 180.0
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
  result = run_pipeline(options=options, plugin_manager=plugin_manager)
  return name + '.pdf'


if __name__ == '__main__':
  cpu = multiprocessing.cpu_count()
  create_files('La Stampa', '/mnt/storage01/sormani')


