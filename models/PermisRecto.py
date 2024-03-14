from models.Permis import Permis
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'
class PermisRecto(Permis):
    def __init__(self,image_path):
        self.image_path=image_path
    def data(self):
            while True:
                img=super().model(image_path=self.image_path)
                #Prenom
                prenom_crp= img[190:240,306:550]
                #nom    
                nom_crp= img[290:330,306:550]
                #cin    
                cin_crp= img[360:395,630:780]
                #Date de naissance    
                ddn_crp= img[350:395,306:490]
                #Categorie  
                ctg_crp= img[540:600,270:610]

                data={
                    'nom':nom_crp,
                    'prenom':prenom_crp,
                    'cin':cin_crp,
                    'ddn':ddn_crp,
                    'ctg':ctg_crp
                }

                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=super().remove_non_alphanumeric(input_string=ocr_text)

                try:
                    data_cin['age']=super().calculate_age(date_string=data_cin.get('ddn'))
                    data_cin['cin']=super().remove_space(input_string=data_cin['cin'])
                except ValueError:
                    pass
                
                return data_cin
    