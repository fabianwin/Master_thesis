import pandas as pd
import numpy as np
import os
from functions import preprocess_tweets, preprocess

"""
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/master_thesis_eisner-master/data/sentiment_data_1500_manual.csv')
eisner_tweets = eisner_tweets.rename(columns={"text": "content"})

#currently 2 different preprocess funciton --> merge to one
#eisner_tweets = preprocess(eisner_tweets)
eisner_tweets = preprocess_tweets(eisner_tweets)

eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Tweet_processing/eisner_tweets.csv', index = False)
"""


#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
classic_coins = ['#BTC', '#ETH']
venture_capital_backed_coins = ['#BNB','#SOL', '#ADA', '#XRP']
community_driven_coins = ['#DOGE', '#SHIB']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins
venture_capital_backed_coins = ['#BNB']

for coin in venture_capital_backed_coins:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'ticker_set_'+coin+".csv"

    df = pd.read_csv(os.path.join(my_path, my_file))
    df = df.dropna(axis=0, how='any', inplace=False)
    df = preprocess(df)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'ticker_set_preprocessed_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))

print("All Ticker sets are preprocessed")
