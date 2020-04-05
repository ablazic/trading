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

# main file to be used on own PC/laptop

if __name__ == '__main__':
    start_time = datetime.utcnow()
    minutes = int(start_time.strftime('%M:%S.%f')[:2])
    
    # Get data from bittrex
    client = BTCClient()
    for bal in client.get_balances():
         print(bal)
    start_time = start_time.timestamp()
    
    # To ensure program runs in first 5 minutes only for any hour
    if minutes > 5:
         print("The script will run after: ", 61-minutes, "minutes")
         sleep((61-minutes)*60)


    # model to make predictions for the next hour
    next_hour_model = load_model('SavedModels/Best_Model_NextHour.hdf5')
    
    encode = {'1': 'BUY', '0': "SELL"}
    i=0
   
    while True:
 
        stock_data = bittrex_data() # bittrex data
        
        # download reddit data for the relevant subreddit
        reddit_data = reddit_mine("CryptoCurrency", datetime.now().timestamp() - 259200, datetime.now().timestamp())
        
        # preprocess the stock & reddit data
        df = preprocess(stock_data, reddit_data)
        
        # Adjust predictions for volatility

        # If there has been upward movement as compared to the previous value - movement is 1 else 0
        df.ix[df['Close'].shift(-1)>df['Close'], 'movement'] = 1
        df.ix[df['Close'].shift(-1) < df['Close'], 'movement'] = 0
        
        # if the movement value is same as previous value - volatility = 0 else 1 i.e. 
        # if there has been an increase or decrease in both volatility is 0
        df.ix[df['movement'].shift(1)==df['movement'], 'volatility'] = 0
        df.ix[df['movement'].shift(1) != df['movement'], 'volatility'] = 1
        
        # Sum volatility for the last 4 hours
        df['historical_volatility'] = df['volatility'].rolling(4).sum()
        
        # Use Keras/ Tensorflow model to pred on the current data, 1 is 1 hour ahead  & 10 - last 10 hours values are considered
        hour_signal = next_prediction(df, next_hour_model, 1, 10)
        
        # If there have been more than 2 fluctuations in last 4 hours hold 
        # (as market is volatile) & likelihood of incorrect predictions is high
        if df['historical_volatility'].iloc[-1]>2:
            signal = 2 # Hold
        else:
            signal = hour_signal[-1][0]

        print("Signal for the next hour is :", signal)

        # Signal is buy/sell signal 1 is buy & 0 is sell
        if signal == 1:
             client.buy(0.001)
             print("Buy order placed on Bittrex")
        elif signal==0:
             client.sell(0.001)
             print("Sell order placed on Bittrex")

        sleep(3600)
