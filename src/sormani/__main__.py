
from src.sormani.sormani import Sormani

if __name__ == '__main__':
  sormani = Sormani('La Stampa', year = 2016, months = [1,2,3,4,5], days = None)
  sormani.create_all_images()


