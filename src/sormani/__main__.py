
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=[x for x in range(1,13)], days=None)
  #sormani = Sormani('Libero', year=2016, months=[x for x in range(1, 13)], days=None)
  sormani = Sormani('Italia Oggi', year=2016, months=None, days=None)
  #sormani.change_all_contrasts(contrast=50, force=True)
  #sormani = Sormani('Il Fatto Quotidiano', year=2016, months=[x for x in range(6, 13)], days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  #sormani.create_all_images()
  #sormani.save_pages_images()
  #sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  sormani.add_pdf_metadata();




