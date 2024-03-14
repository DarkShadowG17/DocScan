from models.Permis import Permis
from models.PermisRecto import PermisRecto
from models.PermisVerso import PermisVerso
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'

class PermisRectoVerso(Permis):
    def __init__(self,recto,verso):
        self.recto=recto
        self.verso=verso
    def data(self):
            while True:
                recto=super().model(self.recto)
                verso=super().model(self.verso)
                pRecto=PermisRecto(image_path=recto)
                pVerso=PermisVerso(image_path=verso)
                data_cin=pRecto.data()
                data_cin.update(pVerso.data())
                return data_cin
            
