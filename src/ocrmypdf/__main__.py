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
from contextlib import suppress
from multiprocessing import set_start_method

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

def get_files(name, root):
  for filedir, dirs, files in os.walk(root + '/' + name):
    for file in files:
      ext = pathlib.Path(file).suffix
      if ext == '.tif' or ext == '.tiff':
        _file = file[: len(file) - len(ext)]
        if not _file + '.pdf' in files:
          start_time = time.time()
          print(f'Working on \'{file}\'.', end = '')
          pdf_name = (to_pdfa(filedir + '/' + _file))
          final_time = time.time()
          print(f' Pdf/a is created with OCR. Creation takes {round(final_time - start_time)} seconds.')

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

def to_pdfa(name):
  _parser, options, plugin_manager = get_parser_options_plugins(None)
  options.author = None
  options.clean = False
  options.clean_final = False
  options.deskew = False
  options.fast_web_view = 1.0
  options.force_ocr = False
  options.image_dpi = 400
  options.input_file = name + '.tif'
  options.jbig2_lossy = False
  options.jbig2_page_group_size = 0
  options.jobs = None
  options.jpeg_quality = 0
  options.keep_temporary_files = False
  options.keywords = None
  options.languages = {'ita'}
  options.max_image_mpixels = 1000.0
  options.optimize = 0
  options.output_file = name + '.pdf'
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
  options.sidecar = None
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
  new_pdf = get_files('La Stampa', '/mnt/storage01/sormani')
  print(new_pdf)

