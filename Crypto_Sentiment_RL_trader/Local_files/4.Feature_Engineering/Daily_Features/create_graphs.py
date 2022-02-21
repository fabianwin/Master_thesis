import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from lppls import LPPLS
import plotly.express as px

def get_lppls_graphs(symbol):
    # read example dataset into df
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'finance_data_'+symbol+".csv"
    date_cols = ["Date"]
    data = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    if symbol == 'ETH' :
        dates = pd.date_range(start="2017-01-01",end="2021-12-31")
        data['Date'] = dates.sort_values(ascending=False)
        data = data.sort_values(by='Date',ascending=True)
    elif symbol == 'BNB' :
        dates = pd.date_range(start="2017-07-05",end="2021-12-31")
        data["Date"] = dates.sort_values(ascending=False)
        data = data.sort_values(by='Date',ascending=True)
    elif symbol == 'XRP' :
        dates = pd.date_range(start="2017-01-01",end="2021-12-31")
        data = data.sort_values(by='Date',ascending=True)
    data.dropna(axis=0, how='any',subset=['Price (Close)'], inplace=True)


    # convert time to ordinal
    #time = [pd.Timestamp.toordinal(dt.strptime(t1, '%d.%m.%y')) for t1 in data['Date']]
    time = data['Date'].apply(lambda x: x.toordinal())


    # create list of observation data
    price = np.log(data['Price (Close)'].values)

    # create observations array (expected format for LPPLS observations)
    observations = np.array([time, price])

    print(np.count_nonzero(np.isnan(observations)))
    print(np.count_nonzero(~np.isnan(observations)))

    # set the max number for searches to perform before giving-up
    # the literature suggests 25
    MAX_SEARCHES = 25

    # instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
    lppls_model = LPPLS(observations=observations)
    print(lppls_model.fit(MAX_SEARCHES))
    print("-----------")
    print()


    # fit the model to the data and get back the params
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)
    lppls_model.plot_fit(symbol)

    if __name__ == '__main__':
        print('compute LPPLS conf scores fresh')
        # compute the confidence indicator
        res = lppls_model.mp_compute_nested_fits(
            workers=12,
            window_size=120,
            smallest_window_size=30,
            outer_increment=1,
            inner_increment=5,
            max_searches=25,
            # filter_conditions_config={} # not implemented in 0.6.xworkers=CPU_CORES,
        )
        res_df = lppls_model.compute_indicators(res)
        res_df['time'] = [pd.Timestamp.fromordinal(int(t1)) for t1 in res_df['time']]
        res_df.set_index('time', inplace=True)
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests')
        my_file = 'LPPLS_CONF_CSV_'+symbol+".csv"
        res_df.to_csv(os.path.join(my_path, my_file))

        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/LPPLS')
        my_file = 'LPPLS_CONF_CSV_'+symbol+".csv"
        res_df.to_csv(os.path.join(my_path, my_file))
        lppls_model.plot_confidence_indicators(res, symbol)


# should give a plot like the following...

classic_coins = ['BTC', 'ETH']
venture_capital_backed_coins = ['BNB','ADA', 'XRP']
community_driven_coins = ['DOGE']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

for coin in coin_list:
    print(coin)
    get_lppls_graphs(coin)
