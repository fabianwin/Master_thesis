import pandas as pd
import numpy as np
import os
from functions import preprocess_tweets
from functions import preprocess

#################Ticker Sets##################
classic_coins = ['#BTC', '#ETH']
venture_capital_backed_coins = ['#BNB','#SOL', '#ADA', '#XRP']
community_driven_coins = ['#DOGE', '#SHIB']
coin_list_ticker = classic_coins + venture_capital_backed_coins + community_driven_coins

for coin in coin_list_ticker:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'ticker_set_'+coin+".csv"

    df = pd.read_csv(os.path.join(my_path, my_file))
    df = df.dropna(axis=0, how='any', inplace=False)
    df = preprocess(df)
    #df = preprocess_tweets(df)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'ticker_set_preprocessed_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))

print("All Ticker sets are preprocessed")

#################Product Sets##################
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA_INU']
coin_list_product = classic_coins + venture_capital_backed_coins + community_driven_coins

for coin in coin_list_product:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'product_set_'+coin+".csv"

    df = pd.read_csv(os.path.join(my_path, my_file))
    df = df.dropna(axis=0, how='any', inplace=False)
    df = preprocess(df)
    #df = preprocess_tweets(df)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'product_set_preprocessed_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))
