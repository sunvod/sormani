from __future__ import annotations

import pathlib
import cv2
import os
import numpy as np

from PyPDF2 import PdfFileMerger
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from PIL import Image
from pathlib import Path
import numpy as np

from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageTk
import tkinter as tk
from tkinter import Label, Button, RAISED

from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import mnist

from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet_v2 import EfficientNetV2M, EfficientNetV2L
from keras.applications.convnext import ConvNeXtXLarge

import warnings
warnings.filterwarnings("ignore")

class Page:
  def __init__(self, file_name, date, newspaper, original_image, pdf_path, jpg_path, txt_path):
    self.original_file_name = file_name
    self.file_name = file_name
    self.original_image = original_image
    self.original_path = str(Path(original_image).parent.resolve())
    self.date = date
    self.year = date.year
    self.month = date.month
    self.month_text = MONTHS[self.month - 1]
    self.day = date.day
    self.newspaper = Newspaper.create(newspaper.name, original_image, newspaper.newspaper_base, date, newspaper.year, newspaper.number)
    self.pdf_path = pdf_path
    self.pdf_file_name = os.path.join(self.pdf_path, 'pdf', self.file_name) + '.pdf'
    self.jpg_path = jpg_path
    self.txt_path = txt_path
    self.txt_file_name = os.path.join(txt_path, self.file_name) + '.txt'
    self.original_txt_file_name = self.txt_file_name
    self.conversions = []
    self.page_control = -1
  def add_conversion(self, conversion):
    if isinstance(conversion, list):
      for conv in conversion:
        self.conversions.append(conv)
    else:
      self.conversions.append(conversion)
  def set_file_names(self):
    page = ('000' + str(self.newspaper.n_page))[-len(str(self.newspaper.n_pages)) : ]
    #page = ('000' + str(self.newspaper.n_page))[-3:]
    self.file_name = self.newspaper.name.replace(' ', '_') \
                     + '_' + str(self.year) \
                     + '_' + str(self.month_text) \
                     + '_' + (str(self.day) if self.day >= 10 else '0' + str(self.day)) \
                     + '_p' + page
    self.txt_file_name = os.path.join(self.txt_path, self.file_name) + '.txt'
  def change_contrast(self):
    if self.force or not self.isAlreadySeen():
      try:
        contrast = self.contrast if self.contrast is not None else self.newspaper.contrast
        image = Image.open(self.original_image)
        pixel_map = image.load()
        if pixel_map[0, 0] == (64, 62, 22) and pixel_map[image.size[0] - 1, image.size[1] - 1] == (64, 62, 22):
          return False
        image = self._change_contrast(image, contrast)
        pixel_map = image.load()
        pixel_map[0, 0] = (64, 62, 22)
        pixel_map[image.size[0] - 1, image.size[1] - 1] = (64, 62, 22)
        image.save(self.original_image)
        return True
      except:
        pass
    return False
  def _change_contrast(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def save_pages_images(self, storage):
    if self.isAlreadySeen():
      pos = self.newspaper.get_whole_page_location()
      image = Image.open(self.original_image)
      image = image.crop(pos)
      image = image.resize(((int)(image.size[0] * 1.5), (int)(image.size[1] * 1.5)), Image.Resampling.LANCZOS)
      n_files = sum(1 for _, _, files in os.walk(storage) for f in files)
      file_count = str('00000000' + str(n_files))[-7:]
      file_name = os.path.join(storage, file_count + '_' + self.file_name) + pathlib.Path(self.original_image).suffix
      image.save(file_name)
      return True
    return False
  def convert_from_cv2_to_image(self, img: np.ndarray) -> Image:
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)
  def convert_from_image_to_cv2(self, img: Image) -> np.ndarray:
    # return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
    return np.asarray(img)
  def _change_contrast_PIL(self, img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
      return 128 + factor * (c - 128)
    return img.point(contrast)
  def change_contrast_PIL(self, image, level):
    image = self._change_contrast_PIL(image, level)
    return self.convert_from_image_to_cv2(image)
  def change_contrast_cv2(self, img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img
  def cv2_resize(self, img, scale):
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    return img
  def delete_non_black(self, img):
    min = np.array([20, 20, 20], np.uint8)
    max = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(img, min, max)
    img[mask > 0] = [255, 255, 255]
    return img
  def add_boxes(self, images, img, contours, hierarchy, parameters, file_name, no_resize):
    def _get_contours(e):
      return e[0]
    _contours = []
    for i, contour in enumerate(contours):
      x, _, _, _ = cv2.boundingRect(contour)
      _contours.append((x, contour))
    _contours.sort(key = _get_contours)
    for i, (_, contour) in enumerate(_contours):
      x, y, w, h = cv2.boundingRect(contour)
      perimeter = cv2.arcLength(contour, True)  # hierarchy[0, i, 3] == -1 and
      roi = img[y:y + h, x:x + w]
      if w > parameters.box[0] and w < parameters.box[1] and h > \
          parameters.box[2] and h < parameters.box[3]:
        mean = roi.mean()
        if mean >= parameters.min_mean and mean <= parameters.max_mean:
          name = file_name + '_' + ('0000' + str(i + 1))[-5:]
          if not no_resize:
            roi = cv2.resize(roi, NUMBER_IMAGE_SIZE)
          if parameters.internal_box is not None:
            cnts_inside, hierarchy_inside = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            _cnts_inside = []
            for i, cnt_inside in enumerate(cnts_inside):
              _x, _, _, _ = cv2.boundingRect(cnt_inside)
              _cnts_inside.append((_x, cnt_inside))
            _cnts_inside.sort(key=_get_contours)
            for j, (_, cnt_inside) in enumerate(_cnts_inside):
              _x, _y, _w, _h = cv2.boundingRect(cnt_inside)
              _roi = roi[_y:_y + _h, _x:_x + _w]
              name_i = file_name + '_' + ('0000' + str(j + 1))[-5:]
              if (_w > parameters.internal_box[0] and
                 _w < parameters.internal_box[1] and
                 _h > parameters.internal_box[2] and
                 _h < parameters.internal_box[3]):
                if not no_resize:
                  _roi = cv2.resize(_roi, NUMBER_IMAGE_SIZE)
                images.append((name_i, _roi, perimeter))
          else:
            images.append((name, roi, perimeter))
  def get_boxes(self, image, level=200, no_resize=False):
    img = self.change_contrast_PIL(image, level)
    img = self.change_contrast_cv2(img)
    parameters = self.newspaper.get_parameters()
    p = self.file_name.split('_')[-1][1:]
    if parameters.include is not None:
      if not int(p) in parameters.include:
        return None, img
    if parameters.exclude is not None:
      if int(p) in parameters.exclude:
        return None, img
    img = self.cv2_resize(img, parameters.scale)
    # Questo cancella tutto ciò che non è
    if parameters.exclude_colors is not None and len(parameters.exclude_colors) == 3:
      if parameters.exclude_colors[0] >= 0:
        red = np.where(img[:, :, 0] > parameters.exclude_colors[0])
      else:
        red = np.where(img[:, :, 0] < -parameters.exclude_colors[0])
      if parameters.exclude_colors[1] >= 0:
        green = np.where(img[:, :, 1] > parameters.exclude_colors[1])
      else:
        green = np.where(img[:, :, 1] < -parameters.exclude_colors[1])
      if parameters.exclude_colors[2] >= 0:
        blue = np.where(img[:, :, 2] > parameters.exclude_colors[2])
      else:
        blue = np.where(img[:, :, 2] < -parameters.exclude_colors[2])
      black_pixels = (np.concatenate((red[0], green[0], blue[0]), axis = 0),\
                     np.concatenate((red[1], green[1], blue[1]), axis = 0))
      img[black_pixels] = [255, 255, 255]
    bimg = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Questo riempie i buchi
    if parameters.fill_hole is not None:
      thresh, binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
      kernel = np.ones((parameters.fill_hole, parameters.fill_hole), np.uint8)
      binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_DILATE, kernel, iterations=5)
      gray = cv2.morphologyEx(binaryImage, cv2.MORPH_ERODE, kernel, iterations=5)
    img = cv2.threshold(gray, parameters.ts, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
      if cv2.contourArea(c) < parameters.max_fillarea:
        cv2.drawContours(img, [c], -1, (255, 255, 255), -1)
    if not parameters.invert:
      img = 255 - img
    edges = cv2.Canny(img, 1, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
      x, y, w, h = cv2.boundingRect(c)
      cv2.rectangle(bimg, (x, y), (x + w, y + h), (36, 255, 12), 2)
    file_name = Path(self.file_name).stem
    images = [(self.file_name + '_00000', bimg)]
    self.add_boxes(images, img, contours, hierarchy, parameters, file_name, no_resize)
    return images, img
  def get_pages_numbers(self, no_resize = False, filedir = None):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      images, img = self.get_boxes(cropped, no_resize = no_resize)
      if filedir is not None and images is not None:
        for img in images:
          image = Image.fromarray(img[1])
          Path(filedir).mkdir(parents=True, exist_ok=True)
          image.save(os.path.join(filedir, img[0] + '.jpg'), format="JPEG")
        images = None
      return images
    return None
  def get_head(self):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      file_name = Path(self.original_image).stem
      return [[file_name, cropped]]
    return None
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
  def add_pdf_metadata(self, first_number = None):
    if not os.path.isfile(self.pdf_file_name):
      return
    if first_number is None:
      first_number = 0
    try:
      file_in = open(self.pdf_file_name, 'rb')
      os.rename(self.pdf_file_name, self.pdf_file_name + '.2')
      pdf_merger = PdfFileMerger()
      pdf_merger.append(file_in)
      pdf_merger.addMetadata({
        '/Keywords': 'Nome del periodico:' + self.newspaper.name
                     + ' ; Anno:' + str(self.year)
                     + ' ; Mese:' + str(self.month)
                     + ' ; Giorno:' + str(self.day)
                     + ' ; Numero del quotidiano:' + str(int(self.newspaper.number) + first_number)
                     + ' ; Anno del quotidiano:' + self.newspaper.year,
        '/Title': self.newspaper.name,
        '/Nome_del_periodico': self.newspaper.name,
        '/Anno': str(self.year),
        '/Mese': str(self.month),
        '/Giorno': str(self.day),
        '/Data': str(self.newspaper.date),
        '/Pagina:': str(self.newspaper.n_page),
        '/Numero_del_quotidiano': str(self.newspaper.number),
        '/Anno_del_quotidiano': str(self.newspaper.year),
        '/Producer': 'osi-servizi-informatici.cloud - Milano'
      })
      file_out = open(self.pdf_file_name, 'wb')
      pdf_merger.write(file_out)
      file_in.close()
      file_out.close()
      os.remove(self.pdf_file_name + '.2')
    except:
      os.remove(self.pdf_file_name + '.2')
      file_in.write(self.pdf_file_name)
  def check_pages_numbers(self, model):
    if self.isAlreadySeen():
      image = Image.open(self.original_image)
      cropped = self.newspaper.crop_png(image)
      images, img = self.get_boxes(cropped)
      prediction = None
      predictions = None
      if images is not None:
        prediction, predictions = self.get_page_numbers(model, images)
      if images is None or prediction is None:
        self.page_control = -1
      elif prediction == self.newspaper.n_page:
        self.page_control = 1
      else:
        self.page_control = 0
      return images, prediction, predictions
  def get_page_numbers(self, model, images):
    images.pop(0)
    dataset = []
    for image in images:
      img = cv2.cvtColor(image[1], cv2.COLOR_GRAY2RGB)
      img = Image.fromarray(img)
      img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      dataset.append(img)
    try:
      original_predictions = list(np.argmax(model.predict(np.array(dataset), verbose = 0), axis=-1))
    except:
      return None, None
    b = None
    predictions = []
    for e in original_predictions:
      if b is None:
        b = e
        if e == 0:
          continue
      elif e ==  0 and (b == 0 or (b == 10 and  len(predictions) == 0) or len(predictions) >= 2):
        continue
      if e != 10:
        predictions.append(str(e))
      b = e
    predictions = ''.join(predictions)
    if len(predictions):
      prediction = int(predictions)
    else:
      prediction = None
    return prediction, original_predictions
  def open_win_pages_files(self, image, file_to_be_changing, prediction = None):
    def close():
      gui.destroy()
      exit()
    def end():
      global end_flag
      gui.destroy()
      end_flag = True
    def number_chosen(button_press):
      global new_file
      global end_flag
      global flag_digited
      end_flag = False
      if button_press != 'ok':
        new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p'
        new_file += str(button_press)
        label1.config(text=new_file)
        if self.file_name != os.path.basename(new_file) and new_file[-1] != 'p':
          on = self.file_name.split('_')[-1][1:]
          n = ('0000' + new_file.split('_')[-1][1:])[-len(on):]
          new_file = '_'.join(new_file.split('_')[:-1]) + '_p' + n
          ext = pathlib.Path(self.original_image).suffix
          file_to_be_changing.append((os.path.join(self.original_path, self.original_image), os.path.join(self.original_path, new_file) + ext))
          if os.path.isdir(self.pdf_path):
            for filedir, _, files in os.walk(self.pdf_path):
              for file in files:
                ext = pathlib.Path(file).suffix
                if Path(file).stem == self.file_name:
                  file_to_be_changing.append((os.path.join(filedir, file), os.path.join(filedir, new_file) + ext))
      gui.destroy()
    def page_chosen(button_press):
      global next_page
      next_page = int(button_press)
      gui.destroy()
    def on_start_hover(event):
      global label3
      img = image.crop((event.x - 40, event.y - 40, event.x + 40, event.y + 40))
      img = img.resize((480, 480), Image.Resampling.LANCZOS)
      crop = ImageTk.PhotoImage(img)
      label3.configure(image=crop)
      label3.image = crop
    global new_file
    global end_flag
    global next_page
    global label3
    end_flag = False
    next_page = -1
    gui = tk.Tk()
    gui.title('ATTENZIONE ! Se confermi verrà mdificato il nome del file in tutti i formati esistenti: ' + self.file_name)
    w = 2140  # Width
    h = 2140  # Height
    screen_width = gui.winfo_screenwidth()
    screen_height = gui.winfo_screenheight()
    x = (screen_width / 2) - (w / 2)
    y = (screen_height / 2) - (h / 2)
    gui.geometry('%dx%d+%d+%d' % (w, h, x, y))
    gui_frame = tk.Frame(gui)
    gui_frame.pack(fill=tk.X, side=tk.BOTTOM)
    # new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p'
    new_file = self.file_name
    if prediction is not None:
      new_file = '_'.join(self.file_name.split('_')[:-1]) + '_p' + str(prediction)
    label1 = Label(gui, text=new_file, font=('Arial 20 bold'), height = 5)
    label1.pack(padx=(10, 10), pady=(10, 10))
    image = image.resize((1550, 2000), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(image)
    label2 = Label(gui_frame, image=img)
    label2.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    button_frame = tk.Frame(gui_frame)
    button_frame.grid(row=0, column=1, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    if self.newspaper.n_page % 2 == 0:
      crop = ImageTk.PhotoImage(image.crop((0, 0, 240, 240)).resize((480, 480), Image.Resampling.LANCZOS))
    else:
      w, h = image.size
      crop = ImageTk.PhotoImage(image.crop((w - 240, 0, w, 240)).resize((480, 480), Image.Resampling.LANCZOS))
    label3 = Label(button_frame, image=crop)
    label3.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    label2.bind('<Button-1>', on_start_hover)
    button_frame_1 = tk.Frame(button_frame)
    button_frame_1.grid(row=1, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(10, 10))
    n_lines = (self.newspaper.n_pages + 2) // 6 + 2
    buttons = [[0 for x in range(6)] for x in range(n_lines)]
    for i in range(n_lines):
      for j in range(6):
        text = i * 6 + j + 1
        if text == self.newspaper.n_pages + 1:
          text = 'ok'
        elif text > self.newspaper.n_pages:
          break
        pixel = tk.PhotoImage(width=1, height=1)
        buttons[i][j] = tk.Button(button_frame_1,
                                  text=text,
                                  compound="center",
                                  font=('Aria', 14),
                                  height=2,
                                  width=4,
                                  padx=0,
                                  pady=0,
                                  command=lambda number=str(text): number_chosen(number))
        buttons[i][j].grid(row=i, column=j, sticky=tk.W + tk.E, padx=(6, 6), pady=(6, 6))
    button_frame_2 = tk.Frame(button_frame)
    button_frame_2.grid(row=2, column=0, sticky=tk.W + tk.E, padx=(10, 10), pady=(100, 100))
    n_lines = self.newspaper.n_pages // 6 + 1
    buttons = [[0 for x in range(6)] for x in range(n_lines)]
    lst = [x + 1 for x in range(self.newspaper.n_pages)]
    for old_file, new_file in file_to_be_changing:
      if pathlib.Path(old_file).suffix == '.tif':
        on = ''.join(Path(old_file).stem.split('_')[-1])[1:]
        nn = ''.join(Path(new_file).stem.split('_')[-1])[1:]
        if on.isdigit():
          on = int(on)
        if nn.isdigit():
          nn = int(nn)
        if on in lst:
          lst.remove(on)
        lst.append(nn)
        pass
    for i in range(n_lines):
      for j in range(6):
        text = i * 6 + j + 1
        if text > self.newspaper.n_pages:
          break
        pixel = tk.PhotoImage(width=1, height=1)
        buttons[i][j] = tk.Button(button_frame_2,
                                  text=text,
                                  compound="center",
                                  font=('Aria', 14),
                                  height=2,
                                  width=4,
                                  padx=0,
                                  pady=0,
                                  command=lambda number=str(text): page_chosen(number))
        if not text in lst:
          buttons[i][j].config(bg='#f00', fg='#fff')
        buttons[i][j].grid(row=i, column=j, sticky=tk.W + tk.E, padx=(6, 6), pady=(6, 6))
    end_button = Button(button_frame, text="Fine", font=('Arial', 18), command=end, height=2, width=4)
    end_button.grid(row=3, column=0, sticky=tk.W + tk.E, padx=(5, 5), pady=(5, 5))
    exit_button = Button(button_frame, text="Esci", font=('Arial', 18), command=close, height=2, width=4)
    exit_button.grid(row=4, column=0, sticky=tk.W + tk.E, padx=(5, 5), pady=(5, 5))
    gui.mainloop()
    return file_to_be_changing, end_flag, next_page
  def rename_pages_files(self, file_to_be_changing, model = None):
    if self.isAlreadySeen():
      if os.path.isdir(self.pdf_path):
        filedir, dirs, files = next(os.walk(self.pdf_path))
        for dir in dirs:
          if dir != 'pdf':
            image = Image.open(os.path.join(filedir, dir, self.original_file_name + '.jpg'))
            break
      else:
        image = Image.open(self.original_image)
      prediction = None
      if model is not None:
        _, prediction, _ = self.check_pages_numbers(model)
      file_to_be_changing, end_flag, next_page = self.open_win_pages_files(image, file_to_be_changing, prediction = prediction)
    return file_to_be_changing, end_flag, next_page


