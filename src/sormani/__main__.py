
from src.sormani.sormani import *

if __name__ == '__main__':

  sormani = Sormani('Il Sole 24 Ore', year=2016, months=[x for x in range(1,2)], days=None, only_ins=True)
  # sormani.set_giornali_pipeline(no_division=False, no_set_names=False, no_change_contrast=False, no_create_image=False)

  # sormani = Sormani('La Domenica del Corriere', year=1900, months=1, days=[x for x in range(1,2)])
  # sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()
  # sormani.improve_images(limit=200, threshold="b0")
  # sormani.rotate_fotogrammi()
  # sormani.clean_images(limit=100, threshold="b9", debug=False)
  # sormani.clean_images(limit=50, threshold="90", debug=True)

  # sormani.remove_borders(verbose=False)

  # sormani.change_colors(inversion = True, limit = "ba")

  # sormani.change_colors(inversion=False, limit="c0")

  # sormani.select_images(limit=200)

  # sormani.set_fotogrammi_folders()

  # sormani = Sormani('Il Mondo', year=1949, months=2, days=19)

  # sormani.get_pages_numbers()
  # sormani.save_pages_images()
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.convert_all_images()
  # sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, REPOSITORY), pages = None, no_resize = True, save_head = True)
  # sormani.get_crop(filedir=os.path.join(STORAGE_BASE, REPOSITORY), no_resize = False, force=True)
  # sormani.rename_pages_files(do_prediction=False)
  sormani.check_page_numbers(save_images=True, model_path='best_model_DenseNet201')
  # sormani.check_page_numbers(save_images = True, model_path = 'best_model_SimpleCNN', assume_newspaper = True, newspaper_name = None)
  # sormani.update_date_creation()




