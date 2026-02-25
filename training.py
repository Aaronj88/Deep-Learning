import numpy as np
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,Dropout,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#downloading nltk dataset
nltk.download("punkt")
nltk.download("stopwords")

#creating text dataset for summarising
data = {
    "text":[
        "The action scenes were hard to follow",
        "The show was very well animated",
        "The characters were well written",
        "The plot was a little overcomplicated sometimes"
        "There was a lot of character devellopement",
        "The plot was interesting",
        "The last part and the ending were quite sad"
    ],
    "label":[
        "negative",
        "positive",
        "positive",
        "negative",
        "positive",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)
print(df.head())


