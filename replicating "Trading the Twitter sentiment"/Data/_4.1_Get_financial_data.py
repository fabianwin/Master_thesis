import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time


apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')
app = TimeSeries(key = apiKey, output_format = 'pandas')




#Get TSLA data
#----------------
finance_data_short_TSLA, finance_data_short_TSLA_meta_data = app.get_daily_adjusted(symbol = 'TSLA', outputsize = 'full')
finance_data_short_TSLA['date']=finance_data_short_TSLA.index
finance_data_short_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_TSLA.csv', index = False)

data= pd.DataFrame()
i=1

while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'TSLA', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    if i != 1:
        df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'TSLA', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_extended_TSLA = data[1:]
finance_data_extended_TSLA.columns =header
finance_data_extended_TSLA.set_index('time')



finance_data_extended_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_TSLA.csv', index = False)

#Get GM data
#-----------------------
finance_data_short_GM, finance_data_short_GM_meta_data = app.get_daily_adjusted(symbol = 'GM', outputsize = 'full')
finance_data_short_GM['date']=finance_data_short_GM.index
finance_data_short_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_GM.csv', index = False)

data= pd.DataFrame()
i=1

while i <=12:
    slice='year1'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    if i != 1:
        df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

i=1
while i <=12:
    slice='year2'
    slice = slice+'month'+str(i)
    totalData = ts.get_intraday_extended(symbol = 'GM', interval = '60min', slice = slice)
    df = pd.DataFrame(list(totalData[0]))
    df = df.iloc[1: , :]
    data = data.append(df)
    time.sleep(12)
    print(slice)
    i += 1

data.dropna(axis=1, how="all", thresh=None)
header = data.iloc[0]
finance_data_extended_GM = data[1:]
finance_data_extended_GM.columns =header
finance_data_extended_GM.set_index('time')

finance_data_extended_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_GM.csv', index = False)
