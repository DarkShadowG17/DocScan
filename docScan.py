
from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
import numpy as np
import itertools
import math
import cv2 
import pytesseract
from pylsd.lsd import lsd
import os
import re
from datetime import datetime
print(os.getcwd())
pytesseract.pytesseract.tesseract_cmd = '.\\Scripts\\Tesseract-OCR\\tesseract.exe'
class DocScanner():
    

    def __init__(self, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
     
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE  



        

    def filter_corners(self, corners, min_dist=20):
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)          

    def get_corners(self, img):

        lines = lsd(img)

        corners = []
        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)


    def get_contour(self, rescaled_image):
        
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)

            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)

    def cin_nouv(self, image_path):
        while True:
            RESCALED_HEIGHT = 500.0
            

        
            image = cv2.imread(image_path)
           #cv2.imshow('CIN', image)
            assert(image is not None)
            
            ratio = image.shape[0] / RESCALED_HEIGHT
           # ratio = 4/3
            orig = image.copy()
            rescaled_image = imutils.resize(image, height =int(RESCALED_HEIGHT))

            # get the contour of the document
            screenCnt = self.get_contour(rescaled_image)

       
            warped = transform.four_point_transform(orig, screenCnt * ratio)
            warped= cv2.resize(warped,(800,600))
            img=warped.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sharpen = cv2.GaussianBlur(gray, (0,0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)


        
            #Prenom
            cv2.rectangle(warped,(278,127),(550,178),(0,255,0),thickness=2)
            prenom_crp= sharpen[127:178,278:550]
            #nom    
            cv2.rectangle(warped,(285,193),(480,250),(0,255,0),thickness=2)
            nom_crp= sharpen[193:245,285:480]
            #cin    
            cv2.rectangle(warped,(72,491),(230,580),(0,255,0),thickness=2)
            cin_crp= sharpen[500:580,72:230]
            #Date de naissance    
            cv2.rectangle(warped,(456,225),(609,290),(0,255,0),thickness=2)
            ddn_crp= sharpen[225:290,456:609]
            #Date de validite   
            cv2.rectangle(warped,(540 ,520),(670,570),(0,255,0),thickness=2)
            ddv_crp= sharpen[520:570,540:670]
            cv2.imshow('Warped',warped)
        
            data={
                'nom':nom_crp,
                'prenom':prenom_crp,
                'cin':cin_crp,
                'ddn':ddn_crp,
                'ddv':ddv_crp
            }

        #     # save the transformed image
            basename = os.path.basename(image_path)
            pressed_key = cv2.waitKey(20) & 0xFF
            if pressed_key == ord('d'):
                break

            elif pressed_key == ord('s'):
                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=self.remove_non_alphanumeric(ocr_text)
                    cv2.imwrite('output/' + key+".jpg", image)
                try:
                    data_cin['age']=self.calculate_age(data_cin.get('ddn'))
                    data_cin['cin']=self.remove_space(data_cin['cin'])
                except ValueError:
                    pass
                print(data_cin)
                print(self.pcin_rec(image_path))
                

                # Save the transformed image
                print("Processed " + basename)
                cv2.imwrite("output\scanned_7" + ".jpg", warped)
                print("image saved in ")
                print(os.getcwd())
                
                cv2.waitKey(500)
            
    
        cv2.destroyAllWindows()
   

    def cin_anc(self,image_path):
        while True:
            RESCALED_HEIGHT = 500.0
            

        
            image = cv2.imread(image_path)
           #cv2.imshow('CIN', image)
            assert(image is not None)
            
            ratio = image.shape[0] / RESCALED_HEIGHT
           # ratio = 4/3
            orig = image.copy()
            rescaled_image = imutils.resize(image, height =int(RESCALED_HEIGHT))

            # get the contour of the document
            screenCnt = self.get_contour(rescaled_image)

       
            warped = transform.four_point_transform(orig, screenCnt * ratio)
            warped= cv2.resize(warped,(800,600))
            img=warped.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            sharpen = cv2.GaussianBlur(gray, (0,0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)


        
             #Prenom
            cv2.rectangle(warped,(0,150),(200,230),(0,255,0),thickness=2)
            prenom_crp= sharpen[150:230,0:200]
            #nom    
            cv2.rectangle(warped,(0,200),(250,280),(0,255,0),thickness=2)
            nom_crp= sharpen[200:280,0:250]
            #cin    
            cv2.rectangle(warped,(520,440),(820,520),(0,255,0),thickness=2)
            cin_crp= sharpen[440:520,520:820]
            #Date de naissance    
            cv2.rectangle(warped,(120,270),(370,320),(0,255,0),thickness=2)
            ddn_crp= sharpen[270:320,120:370]
            #Date de validite   
            cv2.rectangle(warped,(160,390),(395,435),(0,255,0),thickness=2)
            ddv_crp= sharpen[390:435,160:395]
            
            cv2.imshow('Warped',warped)
        
            data={
                'nom':nom_crp,
                'prenom':prenom_crp,
                'cin':cin_crp,
                'ddn':ddn_crp,
                'ddv':ddv_crp
            }

        #     # save the transformed image
            basename = os.path.basename(image_path)
            pressed_key = cv2.waitKey(20) & 0xFF
            if pressed_key == ord('d'):
                break

            elif pressed_key == ord('s'):
                data_cin={}
                for key,image in data.items():
                    ocr_text = pytesseract.image_to_string(image)
                    data_cin[key]=self.remove_non_alphanumeric(ocr_text)
                    cv2.imwrite('output/' + key+".jpg", image)
                try:
                    data_cin['age']=self.calculate_age(data_cin.get('ddn'))
                    data_cin['cin']= self.remove_space(data_cin['cin'])
                except ValueError:
                    pass
                print(data_cin)

                

                # Save the transformed image
                print("Processed " + basename)
                cv2.imwrite("output\scanned_7" + ".jpg", warped)
                print("image saved in ")
                print(os.getcwd())
                
                cv2.waitKey(500)
            
    
        cv2.destroyAllWindows()
    def pcin_rec(self,img_path):
        img = cv2.imread(img_path)
        crp= img[0:600,0:400]

        gray = cv2.cvtColor(crp, cv2.COLOR_BGR2GRAY )

        haar_cascade = cv2.CascadeClassifier('Scripts\haar_face.xml')
        faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=2)
        if len(faces_rect)==0:
            self.cin_anc(img_path)
        else:
            self.cin_nouv(img_path)
        
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
if __name__ == "__main__":
    im_file_path = 'input\cin_badr1.jpg'

    scanner = DocScanner()

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        scanner.pcin_rec(im_file_path)
    