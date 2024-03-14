from flask import Blueprint
app_views=Blueprint('app_views',__name__,url_prefix='/api/v1')
from api.v1.views.CarteNational import *
from api.v1.views.PermisRecto import *
from api.v1.views.PermisVerso import *
from api.v1.views.PermisRectoVerso import *
