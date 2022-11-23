
from src.sormani.sormani import Sormani
from deeplearning import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=[x for x in range(1,13)], days=None)
  #sormani = Sormani('Libero', year=2016, months=[x for x in range(1, 13)], days=None)
  sormani = Sormani('Italia Oggi', year=2016, months=1, days=2)
  #sormani = Sormani('Il Fatto Quotidiano', year=2016, months=[x for x in range(6, 13)], days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  sormani.create_all_images()
  #sormani.save_pages_images()

  # cnn = cnn()

  #see_images()

  pass



