from __future__ import annotations

import pathlib
import datetime
import re
import cv2

from os import listdir

from src.sormani.page import Page
from src.sormani.page_pool import Page_pool
from src.sormani.system import *
from src.sormani.newspaper import Newspaper

import tensorflow as tf
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.filterwarnings("ignore")

PAGE = 1
ISFIRSTPAGE = 2

class AI():
  def __init__(self,  model_path, type, use=False, save=False):
    assert model_path is not None, 'model_path must have a value'
    self.model_path = model_path
    self.type = type
    self.use = use
    self.save = save
    self.model = None

class AIs():

  def __init__(self,  newspaper_name, ais=[]):
    if isinstance(ais, str):
      ais = [ais]
    self.newspaper_name = newspaper_name
    self.ais = ais
    for ai in ais:
      ai.model_path = os.path.join('models', self.newspaper_name.lower().replace(' ', '_'), ai.model_path)

  def add_ai(self, ai):
    self.ais.append(ai)

  def get_model(self, type):
    for ai in self.ais:
      if ai.type == type:
        if ai.model is None and ai.use:
          ai.model = tf.keras.models.load_model(os.path.join(STORAGE_BASE, ai.model_path))
        return ai
    return None

  def garbage_model(self, type):
    for ai in self.ais:
      if ai.type == type:
        if ai.model is not None and ai.use:
          del ai.model
          gc.collect()
          ai.model = None
          return

