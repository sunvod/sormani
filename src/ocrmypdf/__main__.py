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

def create_pdf(name, root):
  file_pool = []
  count_process = 0
  path_pdf = root + '/' + name + '/pdf'
  path_txt = root + '/' + name + '/pdf/txt'
  Path(path_pdf + '/').mkdir(parents=True, exist_ok=True)
  Path(path_txt + '/').mkdir(parents=True, exist_ok=True)
  start_time = time.time()
  print(f'Starting processing \'{name}\' at {str(datetime.datetime.now())}')
  for filedir, dirs, files in os.walk(root + '/' + name):
    if filedir == path_pdf or filedir == path_txt:
      continue
    for file in files:
      ext = pathlib.Path(file).suffix
      if ext == '.tif' or ext == '.tiff':
        _file = file[: len(file) - len(ext)]
        pdffiles = [f for f in listdir(path_pdf) if isfile(join('pdf', f))]
        if not _file + '.pdf' in pdffiles:
          file_pool.append((filedir + '/' + _file, root + '/' + name))
    if len(file_pool) > 0:
      count_process += len(file_pool)
      with Pool() as mp_pool:
        pdf_name = mp_pool.map(to_pdfa, file_pool)
        #pdf_name = mp_pool.imap_unordered(to_pdfa, file_pool)
        final_time = time.time()
  if count_process:
    print(f' \'{name}\' are created with OCR. Creation of {count_process} process ends at {str(datetime.datetime.now())} and takes {round(final_time - start_time)} seconds.')
  else:
    print(f'Warning: There is no files to process for {name}.')



def totext(filename):
    doc = fitz.open(filename)
    toc = doc.get_toc()
    page = doc[0]
    pix = page.get_pixmap()
    print(pix.height) # 4900x7088
    print(pix.width)
    #pix.save("page-%i.png" % page.number)
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
  create_pdf('La Stampa', '/mnt/storage01/sormani')


