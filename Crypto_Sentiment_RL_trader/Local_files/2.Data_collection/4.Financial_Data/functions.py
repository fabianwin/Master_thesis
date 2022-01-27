import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import requests
import os
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
