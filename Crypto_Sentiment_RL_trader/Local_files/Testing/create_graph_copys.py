import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from lppls import lppls, data_loader
import plotly.express as px

def get_lppls_graphs(symbol):
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'finance_data_'+"BTC"+".csv"
    date_cols = ["Date"]
    btc_data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    print(btc_data.Date.head())

    # read example dataset into df
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'finance_data_'+symbol+".csv"
    date_cols = ["Date"]
    data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    data.dropna(axis=0, how='any',subset=['Price (Close)'], inplace=True)


    # convert time to ordinal
    #time = [pd.Timestamp.toordinal(dt.strptime(t1, '%d.%m.%y')) for t1 in data['Date']]
    time = data['Date'].apply(lambda x: x.toordinal())
    #print(time)

    # create list of observation data
    price = np.log(data['Price (Close)'].values)
    #print(price)

    # create observations array (expected format for LPPLS observations)
    observations = np.array([time, price])
    print(observations.shape)

    """
    # read example dataset into df
    data = data_loader.nasdaq_dotcom()
    time = [pd.Timestamp.toordinal(dt.strptime(t1, '%Y-%m-%d')) for t1 in data['Date']]
    #print(time)
    price = np.log(data['Adj Close'].values)
    #print(price)
    # create observations array (expected format for LPPLS observations)
    observations = np.array([time, price])
    print(observations.shape)
    """

    # set the max number for searches to perform before giving-up
    # the literature suggests 25
    MAX_SEARCHES = 25

    # instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
    lppls_model = lppls.LPPLS(observations=observations)
    print(lppls_model.fit(MAX_SEARCHES))


    # fit the model to the data and get back the params
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)
    lppls_model.plot_fit(symbol)


# should give a plot like the following...

classic_coins = ['BTC', 'ETH']
venture_capital_backed_coins = ['BNB','ADA', 'XRP']
community_driven_coins = ['DOGE']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

trouble_list = ['ETH', 'BNB', 'XRP']
trouble_list = ['ETH']

for coin in trouble_list:
    print(coin)
    get_lppls_graphs(coin)
