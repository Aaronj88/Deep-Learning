import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


model_path = "Handwritten_Number_Prediction/mnist.h5"
img_path = "Handwritten_Number_Prediction/handwritten6.png"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    print("File not found!")


def preprocessing_image():
    img = Image.open(img_path).convert("L") #convert("L") makes the image grayscale
    size = max(img.size)
    white_padding = Image.new("L",(size,size),color=255)
    '''print(img.shape)
    print(white_padding.shape)'''
    resized_img = white_padding.resize((28,28),Image.LANCZOS)
    img_array = np.asarray(resized_img).astype("float32")
    print(img_array.shape)
    img_array/=255.0
    img_array = img_array.reshape(1,28,28,1) #1 batch, 28 width, 28 height, 1 channel
    return resized_img,img_array


def predict_img():
    preview,x = preprocessing_image()
    print(model.predict(x))
    predict_num = model.predict(x)[0]
    predict_num = int(np.argmax(predict_num))
    print("predicted number is:",predict_num)


predict_img()


