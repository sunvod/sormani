
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
  #                   days=[40],
  #                   force=True,
  #                   is_bobina=True)
  # # La Domenica Del Corriere
  # sormani.clean_images(threshold=100, thresh_threshold=50, min_threshold=50)
  # # sormani.remove_dark_border()
  # # sormani.cut_at_written_part(threshold=200)
  # # sormani.divide_at_written_part(var_limit=50, ofset=96, x_ofset=750, threshold=100)
  # # sormani.add_borders()
  # # sormani.rotate_final_frames(threshold=100)


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

  # sormani = Sormani('Scenario',
  #                   # days=[x for x in range(17,32)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   days=[40],
  #                   force=True,
  #                   is_bobina=True)

  # sormani.create_all_images(convert=False)

  # sormani.set_all_images_names()
  # sormani.remove_dark_border()
  # sormani.create_all_images(ocr = False)

  # sormani.clean_images(threshold=80, use_ai=False)
  # sormani.remove_dark_border(threshold=220, limit=50, valid=[True,True,True,True])
  # sormani.remove_fix_border(check=[None, None], limit=[0, 0, 100, 0], max=False, border=[False,False])
  # sormani.set_grayscale()

  # sormani = Sormani('La Domenica Del Corriere',
  #                   days=[60],
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

  # # Scenario merge
  # sormani.set_bobine_merge_images()
  # sormani.set_bobine_select_images()
  # sormani.rotate_frames(threshold=50)
  # sormani.bobine_delete_copies()
  #
  # Scenario
  # sormani.clean_images(threshold=200)
  # sormani.remove_dark_border()
  # sormani.cut_at_written_part(threshold=200)
  # sormani.divide_at_written_part(var_limit=50, ofset=96, x_ofset=750, threshold=100)
  # sormani.add_borders()
  # sormani.rotate_final_frames(threshold=100)

  # Scenario prepare images
  # sormani.set_all_images_names()
  # sormani.create_all_images(ocr = True)

  # La Domenica Del Corriere
  # sormani.clean_images(threshold=180)
  # sormani.clean_images(threshold=100, thresh_threshold=50, min_threshold=150)
  # sormani.cut_at_written_part(threshold=200, var_limit=200, y_range=1000)
  # sormani.divide_at_written_part(var_limit=500)
  # sormani.add_borders()
  # sormani.rotate_final_frames(threshold=100)


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

  # sormani.get_pages_numbers()
  # sormani.create_jpg(converts=[Conversion('jpg_small_1500', 150, 60, 1500), Conversion('jpg_medium_1500', 300, 90, 1500)])
  # sormani.add_pdf_metadata()
  # sormani.convert_all_images()
  # sormani.rename_pages_files(do_prediction=False)
  # sormani.update_date_creation()

  # sormani.change_contrast(contrast=100)


  # sormani = Sormani('Italia Artistica Illustrata',
  #                   # days=[x for x in range(17,32)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   days=[1,2,3],
  #                   force=True,
  #                   is_bobina=True)
# Riviste storiche
#   sormani.clean_images(threshold=180)
# sormani.remove_dark_border()
# sormani.cut_at_written_part(threshold=200)
# sormani.divide_at_written_part(var_limit=50, ofset=96, x_ofset=750, threshold=100)
# sormani.add_borders()
# sormani.rotate_final_frames(threshold=100)

# Riviste storiche prepare images
# sormani.set_all_images_names()
# sormani.create_all_images(ocr = True)

  sormani = Sormani('Gazzetta Illustrata',
                    # days=[x for x in range(17,32)],
                    # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
                    # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
                    days=[1,2],
                    force=True,
                    is_bobina=True)
# # Riviste storiche
#   sormani.remove_dark_border()
sormani.set_all_images_names()
sormani.create_all_images(ocr = False, pdf=True)


#   sormani.clean_images(threshold=180)
#
  # sormani = Sormani('Sfera',
  #                   # days=[x for x in range(17,32)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=True, save=True)],
  #                   # ais=[AI('best_model_isfirstpage_DenseNet201_2', ISFIRSTPAGE, use=False)],
  #                   days=[1],
  #                   force=True,
  #                   is_bobina=True)
#   # Riviste storiche
#   sormani.clean_images(threshold=180)

  # sormani = Sormani('Gazzetta Illustrata',
  #                   days=None,
  #                   force=True,
  #                   is_bobina=True)
  # # Riviste storiche
  # sormani.create_all_images(ocr = False)

  # sormani = Sormani('Italia Artistica Illustrata',
  #                   days=None,
  #                   force=True,
  #                   is_bobina=True)
  # # Riviste storiche
  # sormani.create_all_images(ocr = False)
  #
  # sormani = Sormani('La Fornarina',
  #                   days=None,
  #                   force=True,
  #                   is_bobina=True)
  # # Riviste storiche
  # sormani.create_all_images(ocr = False)

  # sormani = Sormani('Sfera',
  #                   days=None,
  #                   force=True,
  #                   is_bobina=True)
  # # Riviste storiche
  # sormani.create_all_images(ocr = False)

  # sormani = Sormani('Scenario',
  #                   days=[70],
  #                   force=True,
  #                   is_bobina=True)
  # # Riviste storiche
  # sormani.create_all_images(ocr = False, pdf=False)