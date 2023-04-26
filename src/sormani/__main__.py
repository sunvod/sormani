
from src.sormani.sormani import *
from src.sormani.AI import *

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
                    is_frames=True,
                    year=1900,
                    months=3,
                    # days=[x for x in range(1,6)],
                    days=[2,3,4,5],
                    # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
                    ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
                    checkimages=False)
  # sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()

  # sormani.improve_images(limit=200, threshold="b0")

  # sormani.rotate_frames()
  # sormani.remove_borders()
  # sormani.bobine_delete_copies()
  # sormani.clean_images(use_ai=True)
  # sormani.center_block(use_ai=True)
  # sormani.center_block()
  # sormani.divide_image()
  # sormani.center_block(use_ai=True)
  # sormani.rotate_final_frames(angle=5)
  # sormani.remove_single_frames()
  # sormani.center_block()

  sormani.delete_gray_on_borders()
  sormani.remove_single_frames()
  sormani.center_block()
  sormani.center_block()

  # sormani.remove_last_single_frames_2()

  # sormani.create_all_images()

  # sormani.rotate_fotogrammi()                                                   # 2
  # sormani.remove_borders()                                                      # 3
  # sormani.bobine_delete_copies()                                                # 4
  # sormani.remove_frames(threshold=190)                                          # 5
  # sormani.divide_image()                                                        # 6
  # sormani.remove_single_frames()         # 7
  # sormani.clean_images(color=248, threshold=230)                                # 8
  # sormani.remove_last_single_frames(default_frame=(100,100,200,100))            # 9
  # sormani.center_block(color=248, model_path='best_model_DenseNet201', use_ai=False)                                                 # 10
  # sormani.create_all_images()

  # sormani.remove_last_single_frames(default_frame=(300,300,300,300))            # 9

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
  # sormani.update_date_creation()

  # sormani.change_contrast(contrast=100)




