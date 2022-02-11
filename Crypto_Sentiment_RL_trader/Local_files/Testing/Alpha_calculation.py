import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


#----------------------------
def finance_ROC(feature_df, n=2):
    """
    - Returns: Feature_set with an additional column describing the rate of change for each sentiment calculaiton method
    """
    roc_df = feature_df[['Close','Date']]
    roc_df.set_index('Date', inplace=True)
    Shift = roc_df.shift(n - 1)
    Diff = roc_df.diff(n - 1)
    ROC = pd.DataFrame(((Diff / Shift) * 100))
    new_column_name = 'ROC_'+str(n)
    ROC.rename({'Close': new_column_name}, axis=1, inplace=True)
    feature_df = pd.merge(feature_df, ROC, how='left',  left_on="Date", right_index=True)

    return feature_df
#----------------------------
def add_alpha_boolean(coin_df, index_df):
    index_df = pd.merge(index_df, coin_df.loc[:, ["date",'ROC_2']], how='left',  left_on="Date", right_on="date")
    index_df = index_df.drop(['date'], axis=1)
    index_df['alpha_return'] = index_df.ROC_2 - index_df.ROC_2_Index
    index_df['alpha_return_bool'] = np.sign(index_df.alpha_return)
    index_df['alpha_return_bool'].replace({-1: False, 1: True}, inplace=True)

    return index_df
#----------------------------
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
my_file = "BITX_index_fund.csv"
index_data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=["Exchange Date"], dayfirst=True)
index_data_df = index_data_df.rename(columns={"Exchange Date":"Date"})
index_data_df = finance_ROC(index_data_df)
index_data_df = index_data_df.rename(columns={"ROC_2":"ROC_2_Index"})

coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']

for coin in coins:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
    my_file = 'complete_feature_set_'+coin+".csv"
    date_cols = ["date"]
    coin_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    fig1 = go.Figure(data=go.Scatter(x=coin_df.date,y=coin_df['Price (Close)'], name=coin, mode='lines'))
    fig1.update_layout(title={'text':coin, 'x':0.5})
    fig1.show()
    index_df = add_alpha_boolean(coin_df, index_data_df)
    """
    if coin == 'BNB':
        fig1 = go.Figure(data=go.Scatter(x=coin_df.date,y=coin_df['Price (High)'], name=coin, mode='lines'))
        fig1.update_layout(title={'text':coin, 'x':0.5})
        fig1.show()
        index_df = add_alpha_boolean(coin_df, index_data_df)
    else:
        fig1 = go.Figure(data=go.Scatter(x=coin_df.date,y=coin_df['Price (Close)'], name=coin, mode='lines'))
        fig1.update_layout(title={'text':coin, 'x':0.5})
        fig1.show()
        index_df = add_alpha_boolean(coin_df, index_data_df)
    """
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=index_df.Date,y=index_df.Close, name='Index'),secondary_y=False)
    if coin == 'BNB':
        fig2.add_trace(go.Scatter(x=coin_df.date,y=coin_df['Price (High)'], name=coin),secondary_y=True)
    else:
        fig2.add_trace(go.Scatter(x=coin_df.date,y=coin_df['Price (Close)'], name=coin),secondary_y=True)
    fig2.add_trace(go.Bar(x=index_df.Date,y=index_df.alpha_return ,name='Alpha Return'),secondary_y=True)

    fig2.show()
