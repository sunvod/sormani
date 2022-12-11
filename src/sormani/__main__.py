
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=1, days=None)
  # sormani = Sormani('La Stampa', year=2016, months=[x for x in range(2,13)], days=None)
  sormani = Sormani('Il Giornale', year=2016, months=1, days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  #sormani.get_pages_numbers()
  # sormani.add_pdf_metadata()
  #sormani.save_pages_images()
  #sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  #sormani.add_pdf_metadata();
  # sormani.count_pages()
  # sormani.change_all_contrasts()
  # sormani.create_all_images()
  sormani.check_page_numbers(save_images = True, model_path = 'best_model_DenseNet201', assume_newspaper = True)
  # sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, 'repository'), pages = None, no_resize = True)

