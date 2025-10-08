#import libraries
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Flatten,Conv2D,MaxPooling2D
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



#import dataset
df = pd.read_csv("A_Z-Handwritten_Data.csv")
#print(df["0"].values)
labels = df["0"].values
images = df.drop('0',axis=1).values
#(train_images, train_labels), (test_images, test_labels) =  tf.keras.datasets.mnist.load_data()


images = images.reshape(-1,28,28,1)
images = images/255.0

labels = to_categorical(labels,num_classes=26)


X_train,X_test,y_train,y_test = train_test_split(images,labels,test_size=0.2,random_state=8)


'''
#reshaping to 28,28,1 px
train_images = train_images.reshape(train_images.shape[0],28,28,1)
train_labels = train_labels.reshape(train_labels.shape[0],28,28,1)
input_shape = (28,28,1)

#converting non_numerical data
num_classes = 10
test_images = tf.keras.utils.to_categorical(test_images,num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels,num_classes)

train_images = train_images.astype('float32')
train_labels = train_labels.astype('float32')

#normalise
train_images /= 255
train_labels /= 255

'''
#build model
input_shape = (28,28,1)
batch_size = 128
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


#train model
hist = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test))
model.save('letter_prediction.h5')
model.summary()


#evaluate model
score = model.evaluate(X_test,y_test,verbose=0)
print("score loss:",score[0])
print("score accuracy:",score[1])