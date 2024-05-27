from flask import request
from models.PermisVerso import PermisVerso 
from models.Contour import Contour
from api.v1.views import app_views



@app_views.route('/carte-grise-verso',methods=['POST'],strict_slashes=False)
def check_carte_grise_verso():


    input=request.get_json()
    image_str=input['base64']
    cn=Contour()
    image = cn.base64_to_image(base64_str=image_str)
    
       
    if image.any():
        carte=PermisVerso(image_path=image)
        return carte.data(),200