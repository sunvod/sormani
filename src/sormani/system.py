import logging
import multiprocessing
import os
import signal
import sys
from collections.abc import Sequence

import portalocker
from PIL import Image, __version__
from argparse import Namespace
from contextlib import suppress

from ocrmypdf import *
from ocrmypdf.__main__ import sigbus
from ocrmypdf._plugin_manager import *
from ocrmypdf.api import *
from pathlib import Path
import ocrmypdf

import warnings

from src.sormani.cli import plugins_only_parser

log = logging.getLogger('ocrmypdf')

warnings.filterwarnings("ignore")

DEBUG = False
NEWSPAPER = 0
BOBINA = 1
FICHES = 2
# DEBUG = True
MULTIPROCESSING = True
# MULTIPROCESSING = False
ORIGINAL_DPI = 400
UPSAMPLING_DPI = 600
THRESHOLDING=0
N_PROCESSES = 12
N_PROCESSES_SHORT = 4
MONTHS = ['GEN', 'FEB', 'MAR', 'APR', 'MAG', 'GIU', 'LUG', 'AGO', 'SET', 'OTT', 'NOV', 'DIC']
JPG_PDF_PATH = 'JPG-PDF'
IMAGE_PATH = 'TIFF'
IMAGE_ROOT = '/mnt/storage01/sormani'
IMAGE_ROOT_2016 = '/home/sormani/2016/giornali/'
IMAGE_ROOT_2017 = '/home/sormani/2017/giornali/'
STORAGE_BASE = '/home/sunvod/sormani_CNN/'
STORAGE_DL = '/home/sunvod/sormani_CNN/giornali/'
STORAGE_BOBINE = os.path.join(IMAGE_PATH, 'Bobine')
CONTRAST = 50
NUMBER_IMAGE_SIZE = (224, 224)
REPOSITORY = 'repository'
NUMBERS = 'numbers'
NO_NUMBERS = 'no_numbers'
NEWSPAPERS_2016 = ['Alias',
              'Alias Domenica',
              'Avvenire',
              'Il Fatto Quotidiano',
              'Il Foglio',
              'Il Giornale',
              'Il Giorno',
              'Il Manifesto',
              'Italia Oggi',
              'La Gazzetta dello Sport',
              'La Stampa',
              'Libero',
              'Milano Finanza',
              'Osservatore Romano',
              'Tutto Libri',
              'Unita',
              'Il Sole 24 Ore']
NEWSPAPERS_2017 = ['Alias',
              'Alias Domenica',
              'Avvenire',
              'Il Fatto Quotidiano',
              'Il Foglio',
              'Il Giornale',
              'Il Giorno',
              'Il Manifesto',
              'Italia Oggi',
              'La Gazzetta dello Sport',
              'La Repubblica',
              'La Stampa',
              'Libero',
              'Milano Finanza',
              'Osservatore Romano',
              'Tutto Libri',
              'Unita',
              'Il Sole 24 Ore',
              'Corriere della Sera']
BOBINE = ['Il Mondo',
          'Il Secolo Illustrato Della Domenica',
          'Le Grandi Firme',
          'Scenario',
          'La Domenica Del Corriere']
SCORECUTOFF = 0.6
HASHCUTOFF = 1
PAGE = 1
ISFIRSTPAGE = 2

global_count = multiprocessing.Value('I', 0)

# def sigbus(self, *args):
#   raise InputFileError("Lost access to the input file")

def exec_ocrmypdf(input_file, output_file='temp.pdf', sidecar_file='temp.txt', image_dpi=ORIGINAL_DPI, oversample=0, thresholding=0):
  _parser, options, plugin_manager = get_parser_options_plugins(args=[input_file, output_file])
  options.input_file = input_file
  options.output_file = output_file
  options.image_dpi = image_dpi
  options.quiet = True
  # options.languages = ['ita']
  options.color_conversion_strategy = 'RGB'
  log = logging.getLogger('ocrmypdf')
  with suppress(AttributeError, PermissionError):
    os.nice(5)
  if not os.isatty(sys.stderr.fileno()):
    options.progress_bar = False
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
  exit_code = run_pipeline_cli(options=options, plugin_manager=plugin_manager)
  if exit_code == ExitCode.child_process_error or not os.path.isfile(output_file):
    image = Image.open(input_file)
    image.save(output_file, "PDF", resolution=70.0)
    print(f'Warning: \'{Path(output_file).stem}\' non ha l\'OCR (Code error: ' + str(exit_code) + ')\n')
    with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
      sormani_log.write('No OCR: ' + Path(output_file).stem  + '(Code error: ' + str(exit_code) + ')\n')
  # ocrmypdf.ocr(input_file, output_file, language='ita')


# def exec_ocrmypdf(input_file, output_file='temp.pdf', sidecar_file='temp.txt', image_dpi=ORIGINAL_DPI, oversample=0, thresholding=0):
#   pre_options, _ = plugins_only_parser.parse_known_args(args=None)
#   plugin_manager = get_plugin_manager(pre_options.plugins)
#   parser = get_parser()
#   plugin_manager.hook.initialize(plugin_manager = plugin_manager)
#   plugin_manager.hook.add_options(parser=parser)
#   options = Namespace()
#   options.input_file = input_file
#   options.output_file = output_file
#   options.sidecar = sidecar_file
#   options.image_dpi = image_dpi
#   options.oversample = oversample
#   options.author = None
#   options.clean = False
#   options.clean_final = False
#   options.deskew = False
#   options.fast_web_view = 1.0
#   options.force_ocr = False
#   options.jbig2_lossy = False
#   options.jbig2_page_group_size = 0
#   options.jobs = None
#   options.jpeg_quality = 0
#   options.keep_temporary_files = False
#   options.keywords = None
#   options.languages = {'ita'}
#   options.max_image_mpixels = 1000.0
#   options.optimize = 0
#   options.output_type = 'pdfa'
#   options.pages = None
#   options.pdf_renderer = 'auto'
#   options.pdfa_image_compression = 'auto'
#   options.plugins = []
#   options.png_quality = 0
#   options.progress_bar = True
#   options.quiet = False
#   options.redo_ocr = False
#   options.remove_background = False
#   options.remove_vectors = False
#   options.rotate_pages = False
#   options.rotate_pages_threshold = 14.0
#   options.skip_big = None
#   options.skip_text = False
#   options.subject = None
#   options.tesseract_config = []
#   options.tesseract_oem = None
#   options.tesseract_pagesegmode = None
#   options.tesseract_thresholding = thresholding
#   options.tesseract_timeout = 360.0
#   options.title = None
#   options.unpaper_args = None
#   options.use_threads = True
#   options.user_patterns = None
#   options.user_words = None # '//mnt/storage01/sormani/number_dictionary.txt'
#   options.verbose = -1
#   options.continue_on_soft_render_error = True
#   options.tesseract_downsample_above = False
#   options.tesseract_downsample_large_images = False
#   options.color_conversion_strategy = 'RGB'
#   with suppress(AttributeError, PermissionError):
#     os.nice(5)
#   verbosity = options.verbose
#   if not os.isatty(sys.stderr.fileno()):
#     options.progress_bar = False
#   if options.quiet:
#     verbosity = Verbosity.quiet
#     options.progress_bar = False
#   configure_logging(
#     verbosity,
#     progress_bar_friendly=options.progress_bar,
#     manage_root_logger=True,
#     plugin_manager=plugin_manager,
#   )
#   log.debug('sormani %s', __version__)
#   try:
#     check_options(options, plugin_manager)
#   except ValueError as e:
#     log.error(e)
#     return ExitCode.bad_args
#   except BadArgsError as e:
#     log.error(e)
#     return e.exit_code
#   except MissingDependencyError as e:
#     log.error(e)
#     return ExitCode.missing_dependency
#   with suppress(AttributeError, OSError):
#     signal.signal(signal.SIGBUS, sigbus)
#   # exit_code = ocr.run_pipeline(options=options, plugin_manager=None)
#   exit_code = run_pipeline_cli(options=options, plugin_manager=plugin_manager)
#   if exit_code == ExitCode.child_process_error or not os.path.isfile(output_file):
#     image = Image.open(input_file)
#     image.save(output_file, "PDF", resolution=70.0)
#     print(f'Warning: \'{Path(output_file).stem}\' non ha l\'OCR\n')
#     with portalocker.Lock('sormani.log', timeout=120) as sormani_log:
#       sormani_log.write('No OCR: ' + Path(output_file).stem  + '\n')