
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

  # sormani = Sormani('La Domenica Del Corriere',
  #                   # months=None,
  #                   # days=[x for x in range(17,32)],
  #                   days=[x for x in range(1,40)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   checkimages=False,
  #                   # force=True,
  #                   is_bobina=True)

  # sormani = Sormani('Scenario',
  #                   # days=[x for x in range(17,32)],
  #                   days=None,
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   checkimages=False,
  #                   force=True,
  #                   is_bobina=True)
  #
  # sormani.set_grayscale()

  sormani = Sormani('Il Secolo Illustrato Della Domenica',
                    # days=[x for x in range(17,32)],
                    days=None,
                    # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
                    # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
                    checkimages=False,
                    force=False,
                    is_bobina=True)

  sormani.create_all_images(ocr = False)

  # sormani.clean_images(threshold=80, use_ai=False)
  # sormani.remove_dark_border(threshold=220, limit=50, valid=[True,True,True,True])
  # sormani.remove_fix_border(check=[None, None], limit=[0, 0, 100, 0], max=False, border=[False,False])
  # sormani.set_grayscale()

  # sormani = Sormani('Il Secolo Illustrato Della Domenica',
  #                   # months=None,
  #                   # days=[x for x in range(17,32)],
  #                   days=[2],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   checkimages=False,
  #                   # force=True,
  #                   is_bobina=True)

  # sormani.create_all_images(ocr = False)
  # sormani.add_pdf_metadata()

  # sormani = Sormani('Scenario',
  #                   years=[1950],
  #                   months=[1],
  #                   days=[x for x in range(1,13)],
  #                   # days=[1],
  #                   # days=[10],
  #                   # days=[6,7,8,9,10,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29],
  #                   # ais=[AI('best_model_DenseNet201_firstpage_3', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   checkimages=False,
  #                   is_bobina=True)

  # sormani.create_all_images(ocr = False)
  # sormani.remove_dark_border(valid=[True,True,True,True])
  # sormani.remove_fix_border(check=[None, None], limit=[0, 0, 0, 0], max=True, border=[False,False])

  # sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()

  # sormani.bobine_delete_copies()

  # sormani.bobine_delete_copies()

  # sormani.rotate_frames(threshold=50)
  # sormani.remove_borders()
  # sormani.bobine_delete_copies()

  # sormani.improve_images(limit=200, threshold="b0")

  # sormani.rotate_frames()
  # sormani.remove_borders()
  # sormani.bobine_delete_copies()
  # sormani.clean_images(use_ai=False)
  # sormani.center_block(use_ai=True)
  # sormani.center_block()
  # sormani.divide_image()
  # sormani.center_block(use_ai=True)
  # sormani.remove_single_frames()
  # sormani.center_block()

  # sormani.rotate_final_frames(angle=5)

  # tentativo di pulire il bordo superiore La Domenica del Corriere
  # sormani.delete_gray_on_borders()
  # sormani.remove_single_frames()
  # sormani.center_block()
  # sormani.center_block()

  # tentativo di pulire il bordo superiore Il Mondo
  # sormani.rotate_frames()
  # sormani.remove_dark_border(valid=[False,False,True,True])
  # sormani.divide_image()
  # sormani.bobine_delete_copies()
  # sormani.remove_dark_border(valid=[True,True,False,False])
  # sormani.clean_images(last_threshold=None)
  # sormani.bobine_delete_copies()

  # sormani.remove_fix_border(check=[None, 5450], limit=[0, 0, 0, 100], max=True, border=[False,False])
  # sormani.clean_images()

  # ultimo tentativo di pulire il bordo superiore La DOmenica del Corriere
  # sormani.remove_last_single_frames_2()
  # sormani.center_block(use_ai=True)
  # sormani.remove_single_frames(valid=[True,False,False,False])

  # sormani.set_all_images_names()
  # sormani.create_all_images(ocr = False)

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




