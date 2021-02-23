import config
from flask import Flask,request,render_template
from flask_cors import CORS
from app.controllers.mnist import mnistCtrl 
from app.controllers.tf_ocr import tfocrCtrl

app=Flask(__name__)
CORS(app)
app.config.from_object(config) # 由config.py管理環境變數

app.register_blueprint(mnistCtrl, url_prefix='/mnist')
app.register_blueprint(tfocrCtrl, url_prefix='/tfocr')