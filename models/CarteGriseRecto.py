from models.Permis import Permis
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'
class CGRecto(Permis):
    def __init__(self,image_path):
        self.image_path=image_path
    def data(self):
            while True:
                img=super().model(image_path=self.image_path)
                #Immatricule
                immatricule_crp= img[190:240,306:550]
                #Date    
                date_crp= img[290:330,306:550]
                #Date de validite    
                ddv_crp= img[360:395,630:780]

                data={
                    'Immatricule':immatricule_crp,
                    'Date':date_crp,
                    'ddv':ddv_crp
                }

                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=super().remove_non_alphanumeric(input_string=ocr_text)
                
                return data_cin
    