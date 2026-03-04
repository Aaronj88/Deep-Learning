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
import re
import ssl

#downloading nltk dataset
#nltk.download("punkt")
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download("stopwords")

#creating text dataset for summarising
data = {
    "text":[
        "The action scenes were hard to follow",
        "The show was very well animated",
        "The characters were well written",
        "The plot was a little overcomplicated sometimes",
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

#cleaning the dataset

#first: make it all lowercase
#second: remove punctuation
#third: removing stopwords
stopwords = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]","",text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    return "".join(tokens)

#apply a function to every line of a column
df["cleaned text"] = df["text"].apply(clean_text)

#encoding the labels
le = LabelEncoder()
df["labels encoded"] = le.fit_transform(df["label"])
print(df.head())

#tokenization and padding
tokenizer = Tokenizer()
tokenizer.fit_transform(df["cleaned text"])
text_sequence = tokenizer.texts_to_sequences(df["cleaned text"])
print("after sequencing")
print(df.head())
max_length = 25
X = pad_sequences(text_sequence,maxlen = max_length)
y = df["labels encoded"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)