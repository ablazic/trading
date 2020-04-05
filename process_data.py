import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import timedelta

def preprocess(stock_data, news):
    stock_data.rename(
        columns={'BV': 'Volume (Currency)', 'C': 'Close', 'H': 'High', 'L': 'Low', 'O': 'Open', 'T': 'Timestamp',
                 'V': 'Volume (BTC)'}, inplace=True)

    # Processing stock data next
    stock_data['Timestamp'] = pd.to_datetime(stock_data['Timestamp'])
    stock_data['date'] = stock_data['Timestamp'].dt.date
    stock_data['time'] = stock_data['Timestamp'].dt.time
    stock_data['hour'] = stock_data['Timestamp'].dt.hour
    #
    stock_data = stock_data[['date', 'hour', 'Open', 'High', 'Low', 'Close', 'Volume (BTC)', 'Volume (Currency)']]

    # stock_data.dropna(axis=0, inplace=True)
    print(stock_data.isnull().sum(axis=0))

    # Processing news data first. Moving news 1 hour ahead as we are collapsing everything to previous hour
    news['created_utc'] = pd.to_datetime(news['created_utc'], unit='s')
    news['created_utc'] = news['created_utc'] + timedelta(hours=1)
    news['date'] = news['created_utc'].dt.date
    news['time'] = news['created_utc'].dt.time
    news['hour'] = news['created_utc'].dt.hour
    
    # Groupby date/ hour to get number of topics posted to get activity variable
    news_activity = news.groupby(['date', 'hour'])['created_utc'].count().reset_index()
    news_activity.columns.values[2] = "activity"
    news_activity['date'] = pd.to_datetime(news_activity['date'])
    
    # Drop irrelevant variables
    news.drop(['approved_at_utc', 'banned_at_utc', 'distinguished', 'num_reports'], axis=1, inplace=True)
    news = news[['title', 'created_utc', 'date', 'time', 'hour']]
    
    # Get polarity of each of the titles posted
    news['polarity'] = news['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    # Get subjectivity of each of the title posted
    news['subjectivity'] = news['title'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    
    # Keep only sentiments/ data for relevant titles which have the following keywords 
    news['contains'] = (
    news['title'].str.contains('exchange', case=False) | news['title'].str.contains('crypto', case=False) | news[
        'title'].str.contains('Bitcoin', case=False) | news['title'].str.contains('BTC', case=False) | news[
        'title'].str.contains('technology', case=False) | news['title'].str.contains('develop', case=False) | news[
        'title'].str.contains('ban', case=False) | news['title'].str.contains('allow', case=False))
    
    news = news.ix[news['contains'] > 0, :] 
    news = news.ix[((news['subjectivity'] != 0) & (news['polarity'] != 0)), :]
    agg = news.groupby(['date', 'hour'])['polarity', 'subjectivity'].mean().reset_index()
    
    agg['date'] = pd.to_datetime(agg['date'])
    
    # Moving on to stock data obtained from bittrex
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    
    # join all the titles for a given data & hour
    df = news.groupby(['date', 'hour'])['title'].apply(lambda x: " ".join(x)).reset_index()
    
    # merge news & stock data on date & hour
    df = pd.merge(df, stock_data, on=['date', 'hour'], how='left')
    # get news activity for the last n days for which we've extracted reddit data
    df = pd.merge(df, news_activity, on=['date', 'hour'], how='left')
    df = pd.merge(df, agg, on=['date', 'hour'], how='left')
    # df.dropna(axis=0, inplace=True)
    print(df.isnull().sum(axis=0))
    df['title'].fillna('no_news', inplace=True)
    df = df.fillna(0)
    
    return df.ix[df['Open']!=0.0, :]


