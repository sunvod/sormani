
from src.sormani.sormani import Sormani

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year = 2016, month = 10, day = 1)
  sormani = Sormani('La Stampa', year=2016, month=1, day=4)
  #sormani.create_jpg()
  #sormani.change_all_contrasts()
  sormani.create_all_images()


