
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Giornale', year=2016, months=[x for x in range(2,12)], days=None)
  sormani = Sormani('Libero', year=2016, months=8, days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  sormani.create_all_images()
  #sormani.save_pages_images()
  #sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  #sormani.add_pdf_metadata();

