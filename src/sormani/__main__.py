
from src.sormani.sormani import *
from src.sormani.AI import *

if __name__ == '__main__':

  sormani = Sormani('Libero',
                    checkimages=False,
                    days=[14,21,28],
                    months=1,
                    years=2016,
                    is_bobina=False)

  # sormani.set_all_images_names()
  sormani.create_all_images()






