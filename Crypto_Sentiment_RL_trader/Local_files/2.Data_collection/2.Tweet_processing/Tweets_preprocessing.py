import pandas as pd
import numpy as np
from functions import preprocess_tweets, preprocess

"""
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/master_thesis_eisner-master/data/sentiment_data_1500_manual.csv')
eisner_tweets = eisner_tweets.rename(columns={"text": "content"})

#currently 2 different preprocess funciton --> merge to one
#eisner_tweets = preprocess(eisner_tweets)
eisner_tweets = preprocess_tweets(eisner_tweets)

eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Tweet_processing/eisner_tweets.csv', index = False)
"""


#currently 2 different preprocess funciton --> merge to one
btc_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/ticker_set_#BTC.csv')
btc_df = btc_df.dropna(axis=0, how='any', inplace=False)
btc_df = preprocess(btc_df)

btc_df.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Tweet_processing/ticker_set_#BTC.csv', index = False)
