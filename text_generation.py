import numpy
from tensorflow.keras.utils import to_categorical

#importing training models from RNN_model.py
from RNN_model import (
    create_rnn_model,
    create_lstm_model,
    char_to_idx,
    idx_to_char,
    num_chars,
    word_len
)

#function to generate text
def generate_text(model,seed_text,length=50):
    text = seed_text
    for i in range(length):
        X = [char_to_idx[c] for c in text[-word_len:]]
        X = to_categorical([X],num_classes=num_chars)



