import argparse
import json
import numpy as np
import requests
from PIL import Image
from cv2 import imread, resize
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array as img_to_array
from tensorflow.keras.preprocessing.image import load_img as load_img


#img = imread('./dataset/validation/snoekbaars/snoekbaars381.jpg')
#img = resize(img, (224,224),3)
#data = np.array(img.astype(np.float).reshape(-1, 224, 224, 3))
image_file = './dataset/validation/snoekbaars/snoekbaars381.jpg'

image = img_to_array(load_img(image_file, target_size=(224, 224, 3))) / 255.

image = image.reshape((-1,) + image.shape)
image = image.astype('float16')

print(image.shape)

data = json.dumps({
    "instances":image.tolist()
})
headers = {"content-type": "application/json"}

resp = requests.post('http://192.168.2.20:8501/v1/models/fish_finder:predict', data=data, headers=headers)

print('response.status_code: {}'.format(resp.status_code))     
print('response.content: {}'.format(resp.content))



