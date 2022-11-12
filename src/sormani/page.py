from __future__ import annotations

import pathlib
import time
import datetime

from PIL import Image
from pathlib import Path
from argparse import Namespace
from multiprocessing import Pool
from os import listdir
from PyPDF2 import PdfFileMerger
from ocrmypdf._plugin_manager import get_parser_options_plugins

from src.sormani.system import *
from src.sormani.newspaper import Newspaper


class Page:
  def __init__(self, file_name, date, newspaper, original_image, pdf_path, jpg_path, txt_path):
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
    self.newspaper = Newspaper.create(newspaper.newspaper_base, original_image, newspaper.name, date, newspaper.year, newspaper.number)
    self.pdf_path = pdf_path
    self.pdf_file_name = os.path.join(self.pdf_path, 'pdf', self.file_name) + '.pdf'
    self.jpg_path = jpg_path
    self.txt_path = txt_path
    self.txt_file_name = os.path.join(txt_path, self.file_name) + '.txt'
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
  def change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def isAlreadySeen(self):
    l = len(self.newspaper.name)
    f = self.file_name
    return f[: l] == self.newspaper.name.replace(' ', '_') and \
        f[l] == '_' and\
        f[l + 1: l + 5].isdigit() and \
        f[l + 5] == '_' and\
        f[l + 6: l + 9] in MONTHS and\
        f[l + 9] == '_' and\
        f[l + 10: l + 12].isdigit() and\
        f[l+12] == '_' and\
        f[l+13] == 'p' and\
        f[l  + 14: l + 16].isdigit()
  def extract_page(self):
    return self.newspaper.get_page()

class Page_pool(list):
  def __init__(self, newspaper_name, date, force = False):
    self.newspaper_name = newspaper_name
    self.date = date
    self.force = force
  def set_pages(self, pages):
    self.sort(key = self._set_pages_sort)
    n_pages = len(self) if pages is None or pages > len(self) else pages
    for n_page, page in enumerate(self):
      page.newspaper.n_pages = n_pages
      page.newspaper.n_real_pages = len(self)
      page.newspaper.set_n_page(n_page, self.date)
    self.sort(key=self._n_page_sort)
  def _set_pages_sort(self, page):
    return page.original_file_name
    #return os.path.getmtime(Path(page.original_image))
  def _n_page_sort(self, page):
    return page.newspaper.n_page
  def change_contrast(self, contrast, force = False):
    count = 0
    for page in self:
      if not force and page.isAlreadySeen():
        continue
      contrast = contrast if contrast is not None else page.newspaper.contrast
      image = Image.open(page.original_image)
      image = page.change_contrast(image, contrast)
      image.save(page.original_image)
      count += 1
    return count
  def create_pdf(self, pages = None):
    if len(self):
      start_time = time.time()
      print(f'Starting creating pdf/a of \'{self.newspaper_name}\' of {str(self.date.strftime("%d/%m/%Y"))} at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
      self.set_pages(pages)
      with Pool(processes=N_PROCESSES) as mp_pool:
        mp_pool.map(self.to_pdfa, self)
      print(f'Creation of {len(self)} pdf/a files for \'{self.newspaper_name}\' ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to process for \'{self.newspaper_name}\'.')
  def to_pdfa(self, page):
    _parser, options, plugin_manager = get_parser_options_plugins(None)
    options = Namespace()
    Path(os.path.join(page.pdf_path, 'pdf')).mkdir(parents=True, exist_ok=True)
    Path(page.txt_path).mkdir(parents=True, exist_ok=True)
    exec_ocrmypdf(page.original_image, page.pdf_file_name, page.txt_file_name, ORIGINAL_DPI, UPSAMPLING_DPI)
    self.add_pdf_metadata(page)
  def add_pdf_metadata(self, page):
    os.rename(page.pdf_file_name, page.pdf_file_name + '.2')
    file_in = open(page.pdf_file_name + '.2', 'rb')
    pdf_merger = PdfFileMerger()
    pdf_merger.append(file_in)
    pdf_merger.addMetadata({
      '/Keywords': 'Nome del periodico:' + page.newspaper.name
                   + ' ; Anno:' + str(page.year)
                   + ' ; Mese:' + str(page.month)
                   + ' ; Giorno:' + str(page.day)
                   + ' ; Numero del quotidiano:' + page.newspaper.number
                   + ' ; Anno del quotidiano:' + page.newspaper.year,
      '/Title': page.newspaper.name,
      '/Nome_del_periodico': page.newspaper.name,
      '/Anno': str(page.year),
      '/Mese': str(page.month),
      '/Giorno': str(page.day),
      '/Data': str(page.newspaper.date),
      '/Pagina:': str(page.newspaper.n_page),
      '/Numero_del_quotidiano': str(page.newspaper.number),
      '/Anno_del_quotidiano': str(page.newspaper.year),
      '/Producer': 'osi-servizi-informatici.cloud - Milano'
    })
    file_out = open(page.pdf_file_name, 'wb')
    pdf_merger.write(file_out)
    file_in.close()
    file_out.close()
    os.remove(page.pdf_file_name + '.2')
    # pdf = pdfx.PDFx(page.pdf_file_name)
    # metadata = pdf.get_metadata()
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
      print(f'Starting converting images of \'{self.newspaper_name}\' at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))}')
      with Pool(processes=14) as mp_pool:
        mp_pool.map(self.convert_image, self)
        print(f'Conversion of {len(self)} images ends at {str(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S"))} and takes {round(time.time() - start_time)} seconds.')
    else:
      print(f'Warning: There is no files to convert for \'{self.newspaper_name}\'.')



class Images_group():

  def __init__(self,  newspaper_base, newspaper_name, filedir, files, get_head = False):
    self.newspaper_name = newspaper_name
    self.filedir = filedir
    self.files = files
    year = ''.join(filter(str.isdigit, filedir.split('/')[-3]))
    month = ''.join(filter(str.isdigit, filedir.split('/')[-2]))
    day = ''.join(filter(str.isdigit, filedir.split('/')[-1]))
    if year.isdigit() and month.isdigit() and day.isdigit():
      self.date = datetime.date(int(year), int(month), int(day))
    else:
      raise NotADirectoryError('Le directory non indicano una data.')
    self.newspaper = Newspaper.create(newspaper_base, os.path.join(filedir, files[0]), self.newspaper_name, self.date)
  def get_page_pool(self, newspaper_name, root, ext, image_path, path_exist, force):
    page_pool = Page_pool(newspaper_name, self.date, force)
    for n_page, file in enumerate(self.files):
      dir_in_filedir = self.filedir.split('/')
      txt_in_filedir = list(map(lambda x: x.replace(image_path, 'txt'), dir_in_filedir))
      dir_in_filedir = list(map(lambda x: x.replace(image_path, 'Jpg-Pdf'), dir_in_filedir))
      if pathlib.Path(file).suffix == '.' + ext:
        _file = Path(file).stem
        files_existing = None
        if os.path.isdir(os.path.join('/'.join(dir_in_filedir), path_exist)):
          files_existing = [f for f in listdir(os.path.join('/'.join(dir_in_filedir), path_exist))]
        if force or files_existing is None or not _file + '.pdf' in files_existing:
          page = Page(_file, self.date, self.newspaper, os.path.join(self.filedir, file), '/'.join(dir_in_filedir), '/'.join(dir_in_filedir), '/'.join(txt_in_filedir))
          page_pool.append(page)
    return page_pool
