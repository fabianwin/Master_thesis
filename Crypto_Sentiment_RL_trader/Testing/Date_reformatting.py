import pandas as pd
import numpy as np
import os
import datetime

symbol="BTC"
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis')
my_file = 'ticker_set_sentiment_#'+symbol+".csv"
date_cols = ["date_short","date_medium"]
ticker_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)
for n,row in ticker_data.iterrows():
    dt = row["date_medium"]
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) # Returns a copy
    d_truncated = datetime.date(dt.year, dt.month, dt.day)
    ticker_data.at[n,'date_short'] =  d_truncated


my_file = 'ticker_set_sentiment_1_#'+symbol+".csv"
ticker_data.to_csv(os.path.join(my_path, my_file))
