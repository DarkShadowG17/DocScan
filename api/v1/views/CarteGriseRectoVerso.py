from flask import request
from models.PermisRectoVerso import PermisRectoVerso
from models.Contour import Contour
from api.v1.views import app_views



@app_views.route('/carte-grise-recto-verso',methods=['POST'],strict_slashes=False)
def check_permis_recto_verso():
    input=request.get_json()
    input_recto=input['recto']
    input_verso=input['verso']
    cn1=Contour()
    image_recto= cn1.base64_to_image(base64_str=input_recto)
    cn=Contour()
    image_verso= cn.base64_to_image(base64_str=input_verso)
    
    if image_recto.any() and image_verso.any():
        carte=PermisRectoVerso(recto=image_recto,verso=image_verso)
        return carte.data() ,200