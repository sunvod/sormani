
from src.sormani.sormani import Sormani

if __name__ == '__main__':
  sormani = Sormani('Il Manifesto', year=2016, months=1, days=2)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  sormani.create_all_images()


