
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=1, days=None)
  sormani = Sormani('Avvenire', year=2016, months=12, days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  #sormani.get_pages_numbers()
  sormani.create_all_images(contrast=False)
  #sormani.save_pages_images()
  #sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  #sormani.add_pdf_metadata();

