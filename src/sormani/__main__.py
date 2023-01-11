
from src.sormani.sormani import *

if __name__ == '__main__':
  # sormani = Sormani('Italia Oggi', year=2016, months=12, days=27)  # [x for x in range(1,13)]
  sormani = Sormani('Il Giorno', year=2016, months=1, days=2, notdivide=True, notcheckimages=True)
  # sormani.get_pages_numbers()
  # sormani.add_pdf_metadata()
  # sormani.save_pages_images()
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.change_all_contrasts()
  # sormani.create_all_images()
  # sormani.convert_all_images()
  sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, REPOSITORY), pages = None, no_resize = True, save_head = True)
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.check_page_numbers(save_images=True, model_path='best_model_DenseNet201')
  # sormani.check_page_numbers(save_images = True, model_path = 'best_model_SimpleCNN', assume_newspaper = True, newspaper_name = None)
  # sormani.update_date_creation()
  # sormani.add_jpg_metadata()
  # sormani.check_jpg(integrate=False)



