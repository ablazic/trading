
from time import time as unix_time, sleep
import praw
import pandas as pd
import csv
from pandas.io.json import json_normalize
from datetime import datetime
import requests

# Extract current data from Reddit & Bittrex

class RedditMiner:
    # Attributes extracted for any defined duration
    attributes = ("approved_at_utc", "banned_at_utc", "created_utc", "distinguished", "downs",
                  "gilded", "num_comments", "num_crossposts", "num_reports", "score", "title",
                  "ups")

    def __init__(self, start, end):
        self.client = praw.Reddit(client_id="",   # Enter your Reddit client id
                                  client_secret="", # Enter your client secret
                                  password="", # Enter your password
                                  user_agent="", # Enter your user agent  
                                  username="") # Enter your reddit username
# http://praw.readthedocs.io/en/latest/getting_started/quick_start.html#get-a-reddit-instance
        
        self.last = unix_time()
        self.start = start
        self.end = end
        self.rows = []

    @property
    def user(self):
        return self.client.user.me()

    def extract(self, sub):
        result = {}
        for attr in self.attributes:
            result[attr] = getattr(sub, attr)
        return result

    def fetch(self, subreddit):
        for submission in subreddit.submissions(start=self.start, end=self.end):
            item = self.extract(submission)
            self.on_item(item)

    def load(self, topic):
        sub = self.client.subreddit(topic)
        self.fetch(sub)
        return self.rows

    def on_item(self, item):
        self.rows.append(item)

# function to mine reddit data - topic - subreddit (e.g. CryptoCurrency) - start - starting point for data extraction
# end = ending point for data extraction
def reddit_mine(topic, start, end):
    miner = RedditMiner(start, end)
    rows = miner.load(topic)
    rows = pd.DataFrame(rows)
    return rows

# function to pull data from bittrex
def bittrex_data():
    r = requests.get("https://bittrex.com/Api/v2.0/pub/market/GetTicks?marketName=USDT-BTC&tickInterval=hour")
    j = r.json()
    df = json_normalize(j, 'result')
    return df


if __name__ == '__main__':
    topic = 'CryptoCurrency'
    print(datetime.now())
    miner = RedditMiner(datetime.now().timestamp()-3600, datetime.now().timestamp())
    rows = miner.load(topic)
    df = pd.DataFrame(rows)
    print(df.head())
    df.to_csv("store_news.csv")

