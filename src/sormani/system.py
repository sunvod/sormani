import os
import sys
from argparse import Namespace
from contextlib import suppress

import signal
import sys
from argparse import Namespace
from contextlib import suppress
from ocrmypdf import __version__, ExitCode, BadArgsError, MissingDependencyError, InputFileError
from ocrmypdf._plugin_manager import get_parser_options_plugins
from ocrmypdf._sync import run_pipeline
from ocrmypdf._validation import check_options, log
from ocrmypdf.api import Verbosity, configure_logging

ORIGINAL_DPI = 400
UPSAMPLING_DPI = 400
N_PROCESSES = 14
MONTHS = ['GEN', 'FEB', 'MAR', 'APR', 'MAG', 'GIU', 'LUG', 'AGO', 'SET', 'OTT', 'NOV', 'DIC']


def sigbus(self, *args):
  raise InputFileError("Lost access to the input file")

def exec_ocrmypdf(input_file, output_file='temp.pdf', sidecar_file='temp.txt', image_dpi=ORIGINAL_DPI,
                  oversample=0):
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
  options.user_words = None # '//mnt/storage01/sormani/number_dictionary.txt'
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
  log.debug('sormani %s', __version__)
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