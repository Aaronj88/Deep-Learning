import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
import yfinance as yf

stock_symbol = "GOOGL"

#downloading the data from the last 5 years
data = yf.download(stock_symbol,period="5y")
closing_price = data[["Close"]]

print(closing_price.head())

#normalising the dataset
scaler = MinMaxScaler(feature_range=(0,1))
scaler_closing_price = scaler.fit_transform(closing_price)

print(scaler_closing_price)

def time_series(data,time_steps=60):
    X = []
    y = []
    for i in range(time_steps,len(data)):
        X.append(data[i-time_steps:i,0])
        y.append(data[i,0])
    
    X = np.array(X)
    y = np.array(y)

    return X,y

X,y = time_series(scaler_closing_price)

train_size = int(len(X)*0.8)
X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


