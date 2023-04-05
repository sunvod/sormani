
from src.sormani.sormani import *

if __name__ == '__main__':

  # sormani = Sormani('Il Sole 24 Ore',
  #                   year=2016,
  #                   months=1,
  #                   days=2,
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
                    months=3,
                    days=[x for x in range(31,32)])
  # sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()

  # sormani.improve_images(limit=200, threshold="b0")

  # sormani.rotate_fotogrammi(threshold=200)                                      # 2
  # sormani.remove_borders(threshold=190, limit=1000)                             # 3
  # sormani.bobine_delete_copies()                                                # 4
  # sormani.remove_frames(threshold=200)                                          # 5
  # sormani.divide_image()                                                        # 6
  # sormani.remove_single_frames(threshold=200, default_frame=(0,50,0,0))         # 7
  sormani.clean_images(color=248, threshold=230)                                # 8
  # sormani.remove_last_single_frames(default_frame=(100,100,200,100))            # 9

  # sormani.create_all_images()

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

  # sormani.change_contrast(contrast=100)




