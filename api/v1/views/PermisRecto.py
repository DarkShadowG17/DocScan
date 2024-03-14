from flask import Flask,request,jsonify,render_template, Request
from models.PermisRecto import PermisRecto 
from models.Contour import Contour
from api.v1.views import app_views



@app_views.route('/permis-recto',methods=['POST'],strict_slashes=False)
def check_permis_recto():


    input=request.get_json()
    image_str=input['base64']
    cn=Contour()
    image = cn.base64_to_image(base64_str=image_str)
    
       
    if image.any():
        permis=PermisRecto(image_path=image)
        return permis.data() ,200