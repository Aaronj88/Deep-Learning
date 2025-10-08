import cv2
import numpy as np
from tensorflow.keras.models import load_model

#loading the trained model for prediction
model = load_model("letter_prediction.h5")

#loading the image for testing
image = cv2.imread("J.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image,(28,28))
image = image/255.0

image = image.reshape(1,28,28,1)
letter_pred = model.predict(image)

pred_label = np.argmax(letter_pred)
print("predicted letter is ",chr(pred_label+65))





