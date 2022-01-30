import pandas as pd
import numpy as np
import os
from functions import construct_sentiment_feature_set
#############Global Parameters###################


#############Functions###########################
def merge(symbol):
    #read dataframes
    #get coin data
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'Coin_data_combined_'+symbol+".csv"
    date_cols = ["Date"]
    coin_data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)

    #get LPPLS data
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/LPPLS')
    my_file = 'LPPLS_CONF_CSV_'+symbol+".csv"
    date_cols = ["time"]
    lppls_data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)
    lppls_data_df.drop(['price', '_fits'], axis=1,inplace=True)

    #get ticker sentiment
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis')
    my_file = 'ticker_set_sentiment_#'+symbol+".csv"
    date_cols = ["date_short"]
    ticker_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)
    ticker_data = construct_sentiment_feature_set(ticker_data, symbol,"ticker")


    #get product sentiment
    my_file = 'product_set_sentiment_'+symbol+".csv"
    product_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)

    #merge DataFrame
    #merged_df = pd.merge(df1, df2, how='left', on='Date')



    #save dataset
    #my_file = 'Coin_data_combined_'+symbol+".csv"
    #merged_df.to_csv(os.path.join(my_path, my_file))

#################Main###########################
merge("BTC")
"""
coins=['ADA', 'BNB', 'BTC','DOGE','ETH', 'XRP']

for coin in coins:
    merge(coin)
"""