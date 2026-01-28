import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM,Dense
from tensorflow.keras.utils import to_categorical


text = "hello friends, machine learning is fun!"

#character mapping
characters = sorted(list(set(text)))
num_chars = len(characters)

char_to_idx = {c:i for i,c in enumerate(characters)}
idx_to_char = {i:c for i,c in enumerate(characters)}

X = []
y = []

'''print(text)
print(characters)
print(char_to_idx)'''

word_len = 5
for  i in range(num_chars-word_len):
    X.append([char_to_idx[c] for c in text[i:i+word_len]])
    y.append([char_to_idx[text[i+word_len]]])

'''
print()
print(X)
print()
print(y)
'''

X = np.array(X)
y = np.array(y)

X = to_categorical(X,num_classes=num_chars)
y = to_categorical(y,num_classes=num_chars)

def create_rnn_model():
    model = Sequential([
        SimpleRNN(64,input_shape=(word_len,num_chars)),
        Dense(num_chars,activation="softmax")
    ])

    model.compile(optimizer="adam",loss="categorical_crossentropy")
    model.fit(X,y, epochs=30, batch_size=16,verbose=1)
    return model

def create_lstm_model():
    lstm_model = Sequential([
        LSTM(64,input_shape=(word_len,num_chars)),
        Dense(num_chars,activation="softmax")
    ])
    model.compile(optimizer="adam",loss="categorical_crossentropy")
    model.fit(X,y,epochs=30,batch_size=16,verbose=1)
    return lstm_model








