import numpy as np
from sklearn.metrics import mean_squared_error
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

#reshaping data for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

#building LSTM model
model = Sequential([
    LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam",loss="mean_squared_error")
model.summary()

model.fit(X_train,y_train,epochs=20,batch_size=32,verbose=1)

preds = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test,preds))

print("error:",error) #error: 0.03082543707246542

predicted_prices = scaler.inverse_transform(preds)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize=(10,5))
plt.plot(actual_prices,label="actual prices")
plt.plot(predicted_prices,label="predicted prices")
plt.title("Google Stocks Predictions")
plt.xlabel("Time Series")
plt.ylabel("Stock Prices")
plt.legend()

plt.show()





