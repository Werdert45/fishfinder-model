import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify

from cv2 import imread, resize
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.keras.preprocessing.image import load_img as load_img

# from flask_cors import CORS

app = Flask(__name__)

# uncomment when using cross platform
# CORS(app)

# Testing URL
@app.route('/fishfinder/predict/', methods=['POST'])

def image_classifier():
    image = img_to_array(load_img(BytesIO(base64.b64decode(request.form['b64'])),target_size=(224, 224, 3))) / 255.

    #image = img_to_array(load_img(image_file, target_size=(224, 224, 3))) / 255.

    image = image.reshape((-1,) + image.shape)
    image = image.astype('float16')
    data = json.dumps({
        "instances":image.tolist()
    })
    headers = {"content-type": "application/json"}


    # Post the image
    resp = requests.post('http://192.168.2.20:9000/v1/models/fishfinder:predict', data=data, headers=headers)


    # Get predictions and format them
    predictions = json.loads(resp.text)

    index_max = np.argmax(predictions.get("predictions"))

    #index_max = np.argmax(predictions[0])

    species = ["alver_blankvoorn", "barbeel", "beekforel", "bittervoorn", "bot", "bronforel", "donaubrasem", "elrits", "fint", "grondels", "grote marene", "gup", "harder_band", "karpers", "kopblei", "kopvoorn", "meerval", "paling", "pos", "regenboogforel", "rietvoorn",
               "serpeling", "sneep", "snoeken", "spiering", "stekel_bermpje", "steur", "vetje", "wijting", "winde", "zalm", "zeeforel", "zeelt", "zonnebaars"]

    return species[index_max]

