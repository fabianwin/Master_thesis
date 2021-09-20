import numpy as np
import pandas as pd
#from Feature_functions import number_of_tweets, daily_average_sentiment, sentiment_volatility, sentiment_momentum
from alpha_vantage.timeseries import TimeSeries
import time


apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')

#Get TSLA data
#----------------
data= pd.DataFrame()
i=1
while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'TSLA', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'TSLA', interval = '60min', slice = slice)
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_TSLA = data[1:]
finance_data_TSLA.columns =header
finance_data_TSLA.set_index('time')

finance_data_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_TSLA.csv', index = False)

#Get GM data
#-----------------------
data= pd.DataFrame()
i=1
while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_GM = data[1:]
finance_data_GM.columns =header
finance_data_GM.set_index('time')

finance_data_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_GM.csv', index = False)
