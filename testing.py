import numpy as np
import pandas as pd
import nltk
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from summa.summarizer import summarize
import re
import ssl

nltk.download("punkt_tab")
nltk.data.path.append("C:/Users/Aaron Jha/AppData/Roaming/nltk_data")

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download("stopwords")

max_len = 20

#loading the model and tokenizer
model = load_model("text_classifier.h5")
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

print("model and tokenizer succesfully loaded")


#text preprocessing
stopwords = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]","",text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    return "".join(tokens)

def predict_statement(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence,maxlen = max_len)
    prediction = model.predict(padded)
    print(prediction)
    prediction_val = prediction[0][0]
    label = "positive" if prediction_val > 0.5 else "negative"
    return label, prediction_val


print(predict_statement("The action scenes were hard to follow"))




