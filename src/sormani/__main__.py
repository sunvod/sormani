
from src.sormani.sormani import *
from src.sormani.AI import *

if __name__ == '__main__':

  sormani = Sormani('Avvenire',
                    checkimages=False,
                    days=None,
                    months=8,
                    years=2016,
                    valid_ins=1,
                    is_bobina=False)

  sormani.create_all_images(ocr=True)






