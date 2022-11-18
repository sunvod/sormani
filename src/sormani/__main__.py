
from src.sormani.sormani import Sormani

if __name__ == '__main__':
  sormani = Sormani('Milano Finanza', year=2016, months=[x for x in range(8,13)], days=None)
  #print(sormani.count_pages())
  #print(sormani.count_number())
  #sormani.create_all_images()
  #sormani.save_pages_images()


