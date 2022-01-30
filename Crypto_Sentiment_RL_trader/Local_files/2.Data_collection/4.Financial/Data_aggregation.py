import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import requests
import os
#############Global Parameters###################


#############Functions###########################
def merge(symbol):
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    #read dataframes
    my_file1 = 'finance_data_'+symbol+".csv"
    my_file2 = 'Exchange_Fees_Network_'+symbol+".csv"
    my_file3 = 'Supply_mining_addresses_'+symbol+".csv"
    date_cols = ["Date"]
    df1 = pd.read_csv(os.path.join(my_path, my_file1), parse_dates=date_cols)
    df2 = pd.read_csv(os.path.join(my_path, my_file2), parse_dates=date_cols)
    df3 = pd.read_csv(os.path.join(my_path, my_file3), parse_dates=date_cols)

    #merge DataFrame
    merged_df = pd.merge(df1, df2, how='left', on='Date')
    merged_df = pd.merge(merged_df, df3, how='left', on='Date')

    #save dataset
    my_file = 'Coin_data_combined_'+symbol+".csv"
    merged_df.to_csv(os.path.join(my_path, my_file))

#################Main###########################

coins=['ADA', 'BNB', 'BTC','DOGE','ETH', 'XRP']

for coin in coins:
    merge(coin)
