from flask import request
from models.CarteGriseRecto import CGRecto 
from models.Contour import Contour
from api.v1.views import app_views



@app_views.route('/carte-grise-recto',methods=['POST'],strict_slashes=False)
def check_carte_grise_recto():


    input=request.get_json()
    image_str=input['base64']
    cn=Contour()
    image = cn.base64_to_image(base64_str=image_str)
    
       
    if image.any():
        carte=CGRecto(image_path=image)
        return carte.data() ,200