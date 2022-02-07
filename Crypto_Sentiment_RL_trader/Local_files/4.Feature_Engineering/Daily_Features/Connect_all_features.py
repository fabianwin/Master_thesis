import pandas as pd
import numpy as np
import os
from functions import construct_sentiment_feature_set, construct_finance_feature_set

#############Functions###########################
def merge(symbol):
    #read dataframes
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/Feature_sets')
    date_cols = ["date"]

    #get ticker sentiment
    my_file = 'scaled_ticker_sentiment_feature_set_'+symbol+".csv"
    my_scaled_file = 'ticker_sentiment_feature_set_'+symbol+".csv"
    ticker_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    scaled_ticker_data = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)


    #get product sentiment
    my_file = 'scaled_product_sentiment_feature_set_'+symbol+".csv"
    my_scaled_file = 'product_sentiment_feature_set_'+symbol+".csv"
    product_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    scaled_product_data = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)


    #get coin data
    my_file = 'scaled_finance_feature_set_'+symbol+".csv"
    my_scaled_file = 'finance_feature_set_'+symbol+".csv"
    date_cols = ["Date"]
    finance_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    scaled_finance_data = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)
    print(symbol)
    print(finance_data.dtypes)
    print("---------")

    #merge DataFrame
    final_df = pd.merge(ticker_data, product_data, how='left', on='date')
    final_df = pd.merge(final_df, finance_data, how='left', left_on='date', right_on='Date')

    scaled_final_df = pd.merge(scaled_ticker_data, scaled_product_data, how='left', on='date')
    scaled_final_df = pd.merge(scaled_final_df, scaled_finance_data, how='left', left_on='date', right_on='Date')

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
    my_file = 'scaled_complete_feature_set_'+symbol+".csv"
    my_scaled_file = 'complete_feature_set_'+symbol+".csv"
    final_df.to_csv(os.path.join(my_path, my_file))
    scaled_final_df.to_csv(os.path.join(my_path, my_scaled_file))



#################Main###########################
coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']


for coin in coins:
    merge(coin)
