from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input as ipp
from tensorflow.keras.applications.vgg19 import preprocess_input as vpp
from tensorflow.keras.applications.resnet50 import preprocess_input as rpp
from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename
import cv2
import copy

app = Flask(__name__)

def model_predict(img_path, model, model_name, disease):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    # if disease =='malaria' and model == 'vgg':
    #x=x/255
    x = np.expand_dims(x, axis=0)
    if model_name == 'vgg':
        x = vpp(x)
    elif model_name =='resnet':
        x = rpp(x)
    else:
        x = ipp(x)
    preds = model.predict(x)
    label=np.argmax(preds, axis=1)
    message = ' ';
    if label==0:
        if disease == 'lung':
            message="Not Infected With Pneumonia"
        else:
            message="Infected With Malaria"
    else:
        if disease == 'lung':
            message="Infected With Pneumonia"
        else:
            message="Not Infected With Malaria"

    return message,preds

def elementwise_mul(a, b):
    """Elementwise multiplication."""
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] *= b[i][j]
    return c

def ncc(patch, template):
    # normalize a small section of image (crop)
    mean_patch = float(sum([float(sum(row))/len(row) for row in patch])/len(patch))
    mean_template = float(sum([float(sum(row)) / len(row) for row in template]) / len(template))

    # subtracting the mean(s) from the patch and the template
    norm_patch = [[a - mean_patch for a in row] for row in patch]
    norm_template = [[a - mean_template for a in row] for row in template]

    # numerator of the formula for NCC
    num_mtx = elementwise_mul(norm_patch, norm_template)
    num = sum([sum(row) for row in num_mtx])

    # first term in the denominator of the formula for NCC
    den_mtx_1 = elementwise_mul(norm_patch, norm_patch)
    den_1 = sum([sum(row) for row in den_mtx_1])

    # second term in the denominator of the formula for NCC
    den_mtx_2 = elementwise_mul(norm_template, norm_template)
    den_2 = sum([sum(row) for row in den_mtx_2])

    # denominator of the formula for NCC
    den = np.sqrt(den_1 * den_2)

    # calculating the NCC value between the patch and the template
    try:
        value = float(num / den)
    except:
        value = 0

    # returning the NCC value
    return value


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        disease = request.form.get('disease')
        model = request.form.get('model')
        f = request.files['file']
        img1 = cv2.imread('./static/assets/lung.jpeg',cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread('./static/assets/malaria.jpeg', cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1,(100,100))
        img2 = cv2.resize(img2,(100,100))
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img3 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img3 = cv2.resize(img3,(100,100))

        matches1_3 = ncc(img1,img3)
        matches2_3 = ncc(img2,img3)
        print(matches1_3,matches2_3)
        if matches1_3>=0.55 and disease == 'malaria':
            return "Please upload malaria cell image"
        elif matches2_3>=0.55 and disease == 'lung':
            return "Please upload lung x-ray image"
        elif matches1_3<0.4 and matches2_3<0.4:
            return "Please upload either lung x-ray or malaria cell image"

        MODEL_PATH_vgg19_lung = 'model_vgg19new.h5'
        MODEL_PATH_resnet_lung  ='model_resnetnew.h5'
        MODEL_PATH_inception_lung  = 'model_inception5.h5'
        MODEL_PATH_vgg19_malaria = 'model_malariavgg19.h5'
        MODEL_PATH_resnet_malaria = 'model_malariaresnet.h5'
        MODEL_PATH_inception_malaria = 'model_malariainception.h5'

        if disease == 'lung' and model == 'vgg':
            modelsel = load_model(MODEL_PATH_vgg19_lung)
        elif disease == 'lung' and model == 'resnet':
            modelsel = load_model(MODEL_PATH_resnet_lung)
        elif disease == 'lung' and model == 'inception':
            modelsel = load_model(MODEL_PATH_inception_lung)
        elif disease == 'malaria' and model == 'vgg':
            modelsel = load_model(MODEL_PATH_vgg19_malaria)
        elif disease == 'malaria' and model == 'resnet':
            modelsel = load_model(MODEL_PATH_resnet_malaria)
        elif disease == 'malaria' and model == 'inception':
            modelsel = load_model(MODEL_PATH_inception_malaria)

        messages, preds = model_predict(file_path, modelsel,model, disease)
        predvalue = preds.tolist()
        if disease == 'lung':
            prob_message_noinfection =   str(predvalue[0][0])
            prob_message_infection =   str(predvalue[0][1])
        else:
            prob_message_noinfection = str(predvalue[0][1])
            prob_message_infection = str(predvalue[0][0])

        resultdisease = {
        "infection":prob_message_infection,
        "noinfection": prob_message_noinfection,
        "message":messages
        }
        return jsonify(resultdisease)
    return None


if __name__ == '__main__':
    app.run(debug=True)
