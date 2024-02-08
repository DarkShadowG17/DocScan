from flask import Flask,request,jsonify,render_template, Request
from models.CarteNational import CarteNational 
from models.Contour import Contour
from api.v1.views import app_views



@app_views.route('/cin',methods=['POST'],strict_slashes=False)
def check():

    # try:
    input=request.get_json()
    image_str=input['base64']
    cn=Contour()
    image = cn.base64_to_image(base64_str=image_str)
    
    print('work')
    # except:
    #     return 'Failed to load image'    
    if image.any():
        cin=CarteNational(image_path=image)
        return cin.pcin_rec() ,200