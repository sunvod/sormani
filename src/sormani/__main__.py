
from src.sormani.sormani import *

if __name__ == '__main__':

  # sormani = Sormani('La Gazzetta dello Sport', year=2017, months=[8], days=None)
  # sormani.set_giornali_pipeline()
  # sormani.add_pdf_metadata()

  sormani = Sormani('La Domenica del Corriere', year=1900, months=1, days=None)
  sormani.set_bobine_merge_images()
  # sormani.change_contrast(contrast=-5)
  # sormani.change_threshold(limit = 140, color = 255)
  # sormani.set_bobine_select_images(remove_merge=False)
  # sormani.rotate_fotogrammi(verbose=True, limit=4000)
  # sormani.divide_image(no_rename=True, is_bobina=True)
  # sormani.create_all_images(thresholding=220)
  # sormani.remove_borders()
  # sormani.change_contrast(contrast=-20)
  # sormani.rotate_fotogrammi(verbose=True, limit=4000)
  # sormani.rotate_page(verbose=True)
  # sormani.set_bobine_pipeline(no_division = False, no_set_names = True, no_change_contrast = True)

  # sormani = Sormani('La Domenica del Corriere', year=1899, months=1, days=8)
  # sormani.set_bobine_images()
  # sormani.set_bobine_merges()
  # sormani.rotate_page(verbose=False, limit=5000)
  # sormani.divide_image(no_rename=True, is_bobina=True)
  # sormani.remove_borders()

  # sormani = Sormani('Il Mondo', year=1949, months=2, days=19)
  # sormani.set_bobine_images()
  # sormani.set_bobine_merges()
  # sormani.rotate_page(verbose=False, limit=5000)
  # sormani.divide_image(no_rename=True, is_bobina=True)
  # sormani.remove_borders()

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




