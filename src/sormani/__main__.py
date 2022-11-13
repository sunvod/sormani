
from src.sormani.sormani import Sormani

if __name__ == '__main__':
  sormani = Sormani('La Stampa', year = 2016, month = 1, day = None)
  sormani.change_all_contrasts()
  sormani.create_all_images()


