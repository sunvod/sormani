
from src.sormani.sormani import *

if __name__ == '__main__':


  # sormani = Sormani('Il Sole 24 Ore',
  #                   year=2016,
  #                   months=[x for x in range(11,12)],
  #                   days=30,
  #                   use_ai=False,
  #                   only_ins=True,
  #                   valid_ins=1,
  #                   model_path='best_model_DenseNet201')
  # sormani.set_giornali_pipeline(divide=False, rename=False, change_contrast=False, create_images=True)
  #
  # sormani.get_pages_numbers(filedir=os.path.join(STORAGE_BASE, REPOSITORY), pages = 23, no_resize = True, save_head = True, force=True, debug=False)
  # sormani.check_page_numbers(save_images=True, print_images=False)

  sormani = Sormani('La Domenica del Corriere',
                    year=1900,
                    months=1,
                    days=[x for x in range(2,3)])
  sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()
  # sormani.bobine_delete_copies()
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
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.convert_all_images()
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.check_page_numbers(save_images=True, model_path='best_model_DenseNet201')
  # sormani.update_date_creation()




