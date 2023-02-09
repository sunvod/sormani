
from src.sormani.sormani import *

if __name__ == '__main__':

  # sormani = Sormani('La Gazzetta dello Sport', year=2017, months=[8], days=None)
  # sormani.set_giornali_pipeline()
  # sormani.add_pdf_metadata()

  sormani = Sormani('La Domenica del Corriere', year=1900, months=3, days=[1])
  sormani.set_bobine_merge_images()
  sormani.set_bobine_select_images()
  sormani.improve_images(limit=200, threshold="80")
  sormani.rotate_fotogrammi(verbose=False)
  sormani.remove_borders(verbose=True)

  # sormani.change_colors(inversion = True, limit = "ba")

  # sormani.change_colors(inversion=False, limit="c0")

  # sormani.select_images(limit=200)

  # sormani.set_fotogrammi_folders()

  # sormani = Sormani('La Domenica del Corriere', year=1900, months=2, days=None)
  # sormani.set_bobine_merge_images()

  # sormani.set_bobine_merge_images()
  # sormani.change_contrast(contrast=-5)
  # sormani.change_threshold(limit = 5, color = 255, inversion = False)
  # sormani.set_bobine_select_images(remove_merge=False, threshold=5)
  # sormani.rotate_fotogrammi(verbose=True, limit=4000)
  # sormani.divide_image(no_rename=True, is_bobina=True)
  # sormani.create_all_images(thresholding=220)
  # sormani.remove_borders()
  # sormani.change_contrast(contrast=-100)
  # sormani.rotate_fotogrammi(verbose=True, limit=6000)
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
  # sormani.get_crop(filedir=os.path.join(STORAGE_BASE, REPOSITORY), no_resize = False, force=True)
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.check_page_numbers(save_images=True, model_path='best_model_DenseNet201')
  # sormani.check_page_numbers(save_images = True, model_path = 'best_model_SimpleCNN', assume_newspaper = True, newspaper_name = None)
  # sormani.update_date_creation()




