import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from lppls import LPPLS

def get_lppls_graphs(symbol):
    # read example dataset into df
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'finance_data_'+symbol+".csv"
    data = pd.read_csv(os.path.join(my_path, my_file))
    data.dropna(axis=0, how='any',subset=['Price (Close)'], inplace=True)

    # convert time to ordinal
    time = [pd.Timestamp.toordinal(dt.strptime(t1, '%d.%m.%y')) for t1 in data['Date']]

    # create list of observation data
    price = np.log(data['Price (Close)'].values)

    # create observations array (expected format for LPPLS observations)
    observations = np.array([time, price])

    # set the max number for searches to perform before giving-up
    # the literature suggests 25
    MAX_SEARCHES = 25

    # instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
    lppls_model = LPPLS(observations=observations)

    # fit the model to the data and get back the params
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)
    lppls_model.plot_fit(symbol)

"""
    # compute the confidence indicator
    res = btc_lppls_model.mp_compute_nested_fits(
        workers=8,
        window_size=120,
        smallest_window_size=30,
        outer_increment=1,
        inner_increment=5,
        max_searches=25,
        # filter_conditions_config={} # not implemented in 0.6.x
    )

    btc_lppls_model.plot_confidence_indicators(res, "BTC")
    # should give a plot like the following...
"""
