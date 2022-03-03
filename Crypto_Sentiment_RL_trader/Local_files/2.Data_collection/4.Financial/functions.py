import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import requests
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#############Global Parameters###################
apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')
app = TimeSeries(key = apiKey, output_format = 'pandas')
cc = CryptoCurrencies(key='TCBN46GY5MD7ASKD', output_format='pandas')

#############Functions###########################
def get_intraday_data(symbol):
    url = f'https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={symbol}&market=USD&interval=15min&outputsize=full&apikey={apiKey}&datatype=csv'
    df = pd.read_csv(url)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Intraday_data')
    my_file = 'intraday_finance_data_#'+symbol+".csv"
    df.to_csv(os.path.join(my_path, my_file))

#------------------------------------------------
def get_daily_data(symbol):
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market=USD&apikey={apiKey}&datatype=csv'
    df = pd.read_csv(url)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'daily_finance_data_#'+symbol+".csv"
    df.to_csv(os.path.join(my_path, my_file))

#------------------------------------------------
if __name__ == "__main__":
    fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=("BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "BITW"))
    d = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)'], name="BTC"),row=1, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ETH.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)'], name="ETH"),
                  row=1, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BNB.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)'], name="BNB"),
                  row=2, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ADA.csv')
    fig.add_trace(go.Scatter(x=df['date'], y=df['Price (Close)'], name="ADA"),
                  row=2, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_XRP.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)'], name="XRP"),
                  row=3, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_DOGE.csv')
    fig.add_trace(go.Scatter(x=df['date'], y=df['Price (Close)'], name="DOGE"),
                  row=3, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data/BITX_index_fund.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Close'], name="BITW"),
                  row=4, col=1)
    fig.update_layout(height=800, width=1000)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        font_family="Times New Roman",
        title_font_family="Times New Roman",
    )
    fig.show()

    fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=("BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "BITW"))
    d = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="BTC"),row=1, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ETH.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="ETH"),
                  row=1, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BNB.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="BNB"),
                  row=2, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_ADA.csv')
    fig.add_trace(go.Scatter(x=df['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="ADA"),
                  row=2, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_XRP.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="XRP"),
                  row=3, col=1)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_DOGE.csv')
    fig.add_trace(go.Scatter(x=df['date'], y=df['Price (Close)']/df['Price (Close)'].shift(1) -1, name="DOGE"),
                  row=3, col=2)
    df = pd.read_csv(r'/Users/fabian/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data/BITX_index_fund.csv')
    fig.add_trace(go.Scatter(x=d['date'], y=df['Close']/df['Close'].shift(1)-1, name="BITW"),
                  row=4, col=1)
    fig.update_layout(height=800, width=1000)
    fig.update_layout(showlegend=False)
    fig.update_layout(
        font_family="Times New Roman",
        title_font_family="Times New Roman",
    )
    fig.show()
