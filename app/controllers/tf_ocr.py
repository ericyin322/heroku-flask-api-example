from flask import Blueprint, request, jsonify, redirect
import app.modules.tf_ocr as ocrmodule
import base64


tfocrCtrl = Blueprint('tfocr',__name__)

@tfocrCtrl.route('', methods=['GET', 'POST'])
def getResult():
    base64Image = ''
    if request.method == 'GET':
        return jsonify({'result': "no image"})

    elif request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        base64Image = str(base64.b64encode(image.read()))
        base64Image = base64Image.split('\'')[1]

    elif request.method == 'POST' and 'image' in request.get_json():
        insertValues = request.get_json()
        base64Image = insertValues['image']
    # return base64Image
    return jsonify({'result': str(ocrmodule.getResult(base64Image)), 'image': base64Image})

