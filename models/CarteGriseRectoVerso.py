from models.Permis import Permis
from models.CarteGriseRecto import CGRecto
from models.CarteGriseVerso import CGVerso
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
                pRecto=CGRecto(image_path=recto)
                pVerso=CGVerso(image_path=verso)
                data_cin=pRecto.data()
                data_cin.update(pVerso.data())
                return data_cin