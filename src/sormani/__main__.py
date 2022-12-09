
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=1, days=None)
  sormani = Sormani('Avvenire', year=2016, months=1, days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  #sormani.get_pages_numbers()
  #sormani.change_all_contrasts()
  # sormani.add_pdf_metadata()
  # sormani.create_all_images()
  #sormani.save_pages_images()
  #sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  #sormani.add_pdf_metadata();
  # sormani.count_pages()
  sormani.check_page_numbers()

