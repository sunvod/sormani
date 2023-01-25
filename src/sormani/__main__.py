
from src.sormani.sormani import *

if __name__ == '__main__':

  sormani = Sormani('La Gazzetta dello Sport', year=2017, months=[1], days=2)
  sormani.set_giornali_pipeline()
  # sormani.add_pdf_metadata()

  # sormani = Sormani('Scenario', year=1932, months=3)
  # sormani.divide_all_image(no_rename=True)
  # sormani.rotate_fotogrammi(verbose=True, limit=1500)

  # sormani.set_bobine_images()
  # sormani.get_pages_numbers()
  # sormani.save_pages_images()
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.convert_all_images()
  # sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, REPOSITORY), pages = None, no_resize = False, save_head = False)
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.check_page_numbers(save_images=True, model_path='best_model_DenseNet201')
  # sormani.check_page_numbers(save_images = True, model_path = 'best_model_SimpleCNN', assume_newspaper = True, newspaper_name = None)
  # sormani.update_date_creation()




