import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D #Convolution: taking subsection("filter") of matrix of pixels and applying mathemamatical equations on it (manipulating the image)
from keras.layers import MaxPooling2D #Maxpooling: taking subsection("filter") and splitting it into groups then taking highest value and removing the rest to eliminate noise.
from keras import backend as K

#IMPORTING DATASET
#dataset
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

print("Train images:", X_train.shape)
print("Train labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

#show img (training)
plt.imshow(X_train[5])
plt.show()


#PREPROCESSING
#reshaping
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
input_shape = (28,28,1)

#converting data to categorical/float values
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test,num_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#averaging out colors
X_train /= 255
X_test /= 255


#BUILDING THE MODEL
batch_size = 128
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


#MODEL TRAINING
hist = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test))
model.save('mnist.h5')
model.summary()


#EVALUATE THE MODEL
score = model.evaluate(X_test,y_test,verbose=0)
print("score loss:",score[0])
print("score accuracy:",score[1])



#HW: make MaxPooling function yourself
