#importing neccesary modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout

#preprocessing the data
imgs_path = "C:\\Users\\aaron\\OneDrive\\Documents\\Coding\\Jetlearn\\DeepLearning\\gtsrb-german-traffic-sign\\Train"
prep_data = "C:\\Users\\aaron\\OneDrive\\Documents\\Coding\\Jetlearn\\DeepLearning\\gtsrb-german-traffic-sign\\processed_data.npz"

classes = 43
if os.path.exists(prep_data):
    print("Loading preprocessed data...")
    npzfile = np.load(prep_data)
    data = npzfile["data"]
    labels = npzfile["labels"]
else:
    print("Processing images...")
    data = []
    labels = []
    for i in range(classes):
        img_path = os.path.join(imgs_path, str(i))
        for img in os.listdir(img_path):
            im = Image.open(os.path.join(img_path, img))
            im = im.resize((30, 30))
            im = np.array(im)
            data.append(im)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)

    np.savez_compressed(prep_data, data=data, labels=labels)
    print("Processing complete. Data saved to your computer.")


path = "C:\\Users\\aaron\\OneDrive\\Documents\\Coding\\Jetlearn\\DeepLearning\\gtsrb-german-traffic-sign\\Train\\10\\00010_00003_00002.png"
img = Image.open(path)
img = img.resize((30, 30))
sr = np.array(img)
plt.imshow(img)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("training shape: ", X_train.shape, y_train.shape)
print("testing shape: ", X_test.shape, y_test.shape)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#preparing the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43,activation="softmax"))


#running the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

epochs = 14
history = model.fit(X_train,y_train,epochs=epochs,batch_size=64,validation_data=(X_test,y_test))

#visualising the results
plt.figure(0)
plt.plot(history.history['accuracy'], label="training accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.title = ("Accuracy")
plt.xlabel("epochs")
plt.ylabel = ("accuracy")
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title = ("Loss")
plt.xlabel("epochs")
plt.ylabel = ("loss")
plt.legend()
plt.show()


from sklearn.metrics import accuracy_score
test = pd.read_csv["gtsrb-german-traffic-sign/Test.csv"]
test_labels = test['ClassId'].values
test_img_path = ".../input/gtsrb-german-traffic-sign"
test_imgs = test['Path'].values
test_data = []
test_labels = []
for img in test_imgs:
    im = Image.open(test_img_path + '/'+ img)
    im = im.resize((30,30))
    im = im.array(im)
    test_data.append(im)


test_data = np.array(test_data)
predictions = model.predict_classes(test_data)
print("accuracy:", accuracy_score(test_labels,predictions))


model.save('traffic_Classifier.h5')