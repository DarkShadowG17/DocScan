from models.Permis import Permis
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'

class CGVerso(Permis):
    def __init__(self,image_path):
        self.image_path=image_path
    def data(self):
            while True:
                img=super().model(image_path=self.image_path)
                #Carburant
                carb_crp= img[380:420,70:250]
                #Puissance fiscale
                puissance_crp= img[380:420,70:250]
                #Chassis
                chassis_crp= img[380:420,70:250]

                data={
                    'Carburant':carb_crp,
                    'Puissance':puissance_crp,
                    'Chassis':chassis_crp
                }

                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=super().remove_non_alphanumeric(input_string=ocr_text)

                return data_cin
            