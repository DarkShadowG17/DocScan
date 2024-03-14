from models.Permis import Permis
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'

class PermisVerso(Permis):
    def __init__(self,image_path):
        self.image_path=image_path
    def data(self):
            while True:
                img=super().model(image_path=self.image_path)
                #Date de validite
                ddv_crp= img[380:420,70:250]

                ocr_text = pytesseract.image_to_string(ddv_crp)
                data=super().remove_non_alphanumeric(ocr_text)
                data_cin={"ddv":data}
                return data_cin
            