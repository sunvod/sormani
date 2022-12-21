
from src.sormani.sormani import *

if __name__ == '__main__':
  #sormani = Sormani('Il Manifesto', year=2016, months=1, days=None)
  sormani = Sormani('Italia Oggi', year=2016, months=1, days=[x for x in range(7,31)])  # [x for x in range(1,13)]
  # sormani = Sormani('Unita', year=2016, months=[x for x in range(2,13)], days=None)
  # sormani = Sormani('Unita', year=2016, months=1, days=2)
  # sormani = Sormani('Milano Finanza', year=2016, months=1, days=[2,9])
  # sormani.get_pages_numbers()
  # sormani.add_pdf_metadata()
  # sormani.save_pages_images()
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.change_all_contrasts()
  # sormani.create_all_images()
  sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, REPOSITORY), pages = None, no_resize = True, save_head = False)
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.check_page_numbers(save_images = True, model_path = 'best_model_DenseNet201', assume_newspaper = False, newspaper_name = None)
  # sormani.update_date_creation()
