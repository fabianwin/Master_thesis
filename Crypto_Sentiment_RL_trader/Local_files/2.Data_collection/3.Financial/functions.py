import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import requests
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from datetime import datetime


#############Global Parameters###################
apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')
app = TimeSeries(key = apiKey, output_format = 'pandas')
cc = CryptoCurrencies(key='TCBN46GY5MD7ASKD', output_format='pandas')

#############Functions###########################
def get_intraday_data(symbol):
    url = f'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=USD&interval=15min&outputsize=full&apikey={apiKey}&datatype=csv'
    df = pd.read_csv(url)
    my_path = os.path.abspath(r'/Users/fabianwinkelmannwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Intraday_data')
    my_file = 'intraday_finance_data_#'+symbol+".csv"
    df.to_csv(os.path.join(my_path, my_file))

#------------------------------------------------
def get_daily_data(symbol):
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={apiKey}&datatype=csv'
    df = pd.read_csv(url)
    my_path = os.path.abspath(r'/Users/fabianwinkelmannwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'daily_finance_data_#'+symbol+".csv"
    df.to_csv(os.path.join(my_path, my_file))

#------------------------------------------------
def plot_finance_graph(symbol, df):
    sns.set()
    d = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    fig.suptitle(symbol, fontsize=16)
    ax1.set_ylabel(' ', color=color)
    ax1.plot(df['date'], df['Price (Close)'], color=color)
    ax1.set_ylabel('Value in USD')
    ax1.set_xlabel('Date')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([0,365,730,1096,1461,1826])
    ax1.set_xticklabels(['2017','2018','2019','2020','2021','2022'])
    fig.tight_layout()
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Graphs')
    my_file = symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    sns.reset_orig()
    plt.show()

#------------------------------------------------
def plot_daily_return_graph(symbol, df):
    sns.set()
    d = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    fig.suptitle(symbol, fontsize=16)
    ax1.set_ylabel(' ', color=color)
    ax1.plot(df['date'], 100*(df['Price (Close)']/df['Price (Close)'].shift(1) -1), color=color)
    ax1.set_ylabel('Daily Percentage Change')
    ax1.set_xlabel('Date')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([0,365,730,1096,1461,1826])
    ax1.set_xticklabels(['2017','2018','2019','2020','2021','2022'])
    fig.tight_layout()
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Graphs')
    my_file = symbol+'_daily_return.png'
    plt.savefig(os.path.join(my_path, my_file))
    sns.reset_orig()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ETH.csv')
    plot_finance_graph("ETH", df)
    plot_daily_return_graph("ETH", df)

    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BNB.csv')
    plot_finance_graph("BNB", df)
    plot_daily_return_graph("BNB", df)

    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ADA.csv')
    plot_finance_graph("ADA", df)
    plot_daily_return_graph("ADA", df)

    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_XRP.csv')
    plot_finance_graph("XRP", df)
    plot_daily_return_graph("XRP", df)

    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_DOGE.csv')
    plot_finance_graph("DOGE", df)
    plot_daily_return_graph("DOGE", df)

    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    plot_finance_graph("BTC", df)
    plot_daily_return_graph("BTC", df)

    sns.set()
    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv', index_col=1, parse_dates=True)
    df = df[-151:]
    fund = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data/BITX_index_fund.csv',index_col=0, parse_dates=True)
    fund = fund[-104:]
    new_df = pd.merge(df, fund, how='left',  left_on="date", right_index=True)
    new_df.fillna(method='ffill',inplace=True)
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('BTC', color=color)
    ax1.plot(new_df.index, new_df['Price (Close)'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("BITW",color=color)
    ax2.plot(new_df.index, new_df["Close"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Graphs')
    my_file = 'BITW-BTC.png'
    plt.savefig(os.path.join(my_path, my_file))
    sns.reset_orig()
    plt.show()
