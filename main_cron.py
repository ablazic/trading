import pandas as pd
import numpy as np
import sys
import os
from Load_data import *
from prediction_files import *
from process_data import *
from keras.models import load_model
from time import sleep
from threading import Thread
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from trades import BTCClient

# Get current utc time
start_time = datetime.utcnow()

# Get data from bittrex (defined in trades.py)
client = BTCClient()

# Load Keras/ Tensorflow model
next_hour_model = load_model('SavedModels/Best_Model_NextHour.hdf5')

# Download latest data from bittrex
stock_data = bittrex_data()

# Download data from reddit for the last 3 days - You may subreddit from here
reddit_data = reddit_mine("CryptoCurrency", datetime.now().timestamp() - 259200, datetime.now().timestamp())

# Combine reddit and stock data
df = preprocess(stock_data, reddit_data)

# Adjust predictions for volatility

# If there has been upward movement as compared to the previous value - movement is 1 else 0
df.ix[df['Close'].shift(-1) > df['Close'], 'movement'] = 1
df.ix[df['Close'].shift(-1) < df['Close'], 'movement'] = 0

# if the movement value is same as previous value - volatility = 0 else 1 i.e. 
# if there has been an increase or decrease in both volatility is 0
df.ix[df['movement'].shift(1) == df['movement'], 'volatility'] = 0
df.ix[df['movement'].shift(1) != df['movement'], 'volatility'] = 1

# Sum volatility for the last 4 hours
df['historical_volatility'] = df['volatility'].rolling(4).sum()

# Use Keras/ Tensorflow model to pred on the current data, 1 is 1 hour ahead  & 10 - last 10 hours values are considered
hour_signal = next_prediction(df, next_hour_model, 1, 10)

# If there have been more than 2 fluctuations in last 4 hours hold 
# (as market is volatile) & likelihood of incorrect predictions is high

if df['historical_volatility'].iloc[-1] > 2:
    signal = 2 # Hold
else:
    signal = hour_signal[-1][0]  

print("Signal for the next hour is :", signal)

# Signal is buy/sell signal 1 is buy & 0 is sell
if signal == 1:
    rate = client.get_bid_ask()[0]['Ask'] # Get current Ask price
    if rate < df['Close'].iloc[-1]: # if ask price is less than predicted price buy
         client.buy(0.001, rate)
    print("Buy order placed on Bittrex at: ", rate)

elif signal==0:
      rate = client.get_bid_ask()[0]['Bid'] # Get current bid price
      if rate > df['Close'].iloc[-1]: # if bid price is greater than predicted price sell
          client.sell(0.001, rate)
    print("Sell order placed on Bittrex")




