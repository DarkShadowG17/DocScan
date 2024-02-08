from datetime import datetime
from pyimagesearch import imutils
from pyimagesearch import transform
import cv2 
import re
from models.Contour import Contour
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'
cnt = Contour()

class CarteNational():
    def __init__(self,image_path):
        self.image_path=image_path
    def cin_nouv(self):
            while True:
                RESCALED_HEIGHT = 500.0
                
                image=self.image_path
                assert(image is not None)
                
                ratio = image.shape[0] / RESCALED_HEIGHT
                orig = image.copy()
                rescaled_image = imutils.resize(image, height =int(RESCALED_HEIGHT))

                # get the contour of the document
                screenCnt = cnt.get_contour(rescaled_image)

        
                warped = transform.four_point_transform(orig, screenCnt * ratio)
                warped= cv2.resize(warped,(800,600))
                img=warped.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sharpen = cv2.GaussianBlur(gray, (0,0), 3)
                sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
        
                #Prenom
                prenom_crp= sharpen[127:178,278:550]
                #nom    
                nom_crp= sharpen[193:245,285:480]
                #cin    
                cin_crp= sharpen[500:580,72:230]
                #Date de naissance    
                ddn_crp= sharpen[225:290,456:609]
                #Date de validite   
                ddv_crp= sharpen[520:570,540:670]

                data={
                    'nom':nom_crp,
                    'prenom':prenom_crp,
                    'cin':cin_crp,
                    'ddn':ddn_crp,
                    'ddv':ddv_crp
                }

                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=self.remove_non_alphanumeric(ocr_text)

                try:
                    data_cin['age']=self.calculate_age(data_cin.get('ddn'))
                    data_cin['cin']=self.remove_space(data_cin['cin'])
                except ValueError:
                    pass
                
                return data_cin
            
    

    def cin_anc(self):
            while True:
                RESCALED_HEIGHT = 500.0
                image=self.image_path
                assert(image is not None)
                
                ratio = image.shape[0] / RESCALED_HEIGHT
                orig = image.copy()
                rescaled_image = imutils.resize(image, height =int(RESCALED_HEIGHT))

                # get the contour of the document
                screenCnt = cnt.get_contour(rescaled_image)

        
                warped = transform.four_point_transform(orig, screenCnt * ratio)
                warped= cv2.resize(warped,(800,600))
                img=warped.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                sharpen = cv2.GaussianBlur(gray, (0,0), 3)
                sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

                #Prenom
                prenom_crp= sharpen[150:230,0:200]
                #nom    
                nom_crp= sharpen[200:280,0:250]
                #cin    
                cin_crp= sharpen[440:520,520:720]
                #Date de naissance    
                ddn_crp= sharpen[270:320,120:370]
                #Date de validite   
                ddv_crp= sharpen[390:435,160:395]

                data={
                    'nom':nom_crp,
                    'prenom':prenom_crp,
                    'cin':cin_crp,
                    'ddn':ddn_crp,
                    'ddv':ddv_crp
                }

                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=self.remove_non_alphanumeric(ocr_text)

                try:
                    data_cin['age']=self.calculate_age(data_cin.get('ddn'))
                    data_cin['cin']= self.remove_space(data_cin['cin'])
                except ValueError:
                    pass

                return data_cin

    def pcin_rec(self):
        
        img=self.image_path
        crp= img[0:600,0:400]

        gray = cv2.cvtColor(crp, cv2.COLOR_BGR2GRAY )

        haar_cascade = cv2.CascadeClassifier('Scripts\haar_face.xml')
        faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=2)
        if len(faces_rect)==0:
            data= self.cin_anc()
        else:
            data = self.cin_nouv()
        return data
        
    def calculate_age(self,date_string):
        try:
            date_object_dot = datetime.strptime(date_string, '%d.%m.%Y')

            current_date = datetime.now()

            age_dot = current_date.year - date_object_dot.year - ((current_date.month, current_date.day) < (date_object_dot.month, date_object_dot.day))

            return age_dot
        except ValueError:
            try:
                date_object_slash = datetime.strptime(date_string, '%d/%m/%Y')
                
                current_date = datetime.now()

                age_slash = current_date.year - date_object_slash.year - ((current_date.month, current_date.day) < (date_object_slash.month, date_object_slash.day))

                return age_slash
            except ValueError:
                return None
    def remove_non_alphanumeric(self,input_string):
        filtered_string = ''.join(char for char in input_string if char.isupper() or char.isdigit() or char in {' ', '.', '/'})
        date=self.extract_date_format(filtered_string)
        if date:
            return date
        return filtered_string

    def remove_space(self,input_string):
        filtered_string = ''.join(char for char in input_string if char.isupper() or char.isdigit())
        return filtered_string
    def extract_date_format(self,input_string):
        date_match = re.search(r'\b(?:\D*)(\d{2}[./]\d{2}[./]\d{4})\b', input_string)
        
        if date_match:
            return date_match.group(1)
        else:
            return None