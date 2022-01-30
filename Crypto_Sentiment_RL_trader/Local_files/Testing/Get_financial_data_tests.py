import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time

#############Global Parameters###################
"""
apiKey = 'TCBN46GY5MD7ASKD'
ts = TimeSeries(key = apiKey, output_format = 'csv')
app = TimeSeries(key = apiKey, output_format = 'pandas')


#############Functions###########################
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange

ts = TimeSeries(key = apiKey, output_format='pandas')
data = ts.get_daily_adjusted('MSFT', outputsize='full')
print(data)

cc = CryptoCurrencies(key=apiKey, output_format='pandas')
data = cc.get_digital_currency_monthly(symbol='BTC', market='USD')
print(data)


cc = CryptoCurrencies(key=apiKey, output_format='pandas')
data, meta_data = cc.get_digital_currency_daily(symbol='BTC', market='CNY')
print(data)
print(meta_data)
"""
#############Functions###########################
def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range
#############Functions###########################
from messari.messari import Messari
messari = Messari('f7c66dc1-8cc1-48a5-a61e-2a81f17b1e5d')
assets = ['bitcoin']
metrics = ['price','txn.fee.avg']
metric = 'price'
"""
txn.fee.avg
txn.cnt
txn.tsfr.val.adj
sply.circ
mcap.circ
reddit.subscribers
iss.rate
mcap.realized
bitwise.volume
txn.tsfr.val.avg
act.addr.cnt
fees.ntv
exch.flow.in.usd.incl
blk.size.byte
txn.tsfr.val.med
exch.flow.in.ntv.incl
exch.flow.out.usd
txn.vol
exch.flow.out.ntv.incl
exch.flow.out.usd.incl
txn.fee.med
min.rev.ntv
exch.sply.usd
diff.avg
daily.shp
txn.tsfr.cnt
exch.flow.in.ntv
new.iss.usd
mcap.dom
daily.vol
reddit.active.users
exch.sply
nvt.adj
exch.flow.out.ntv
min.rev.usd
bitwise.price
new.iss.ntv
blk.size.bytes.avg
hashrate
exch.flow.in.usd
price
real.vol
"""
start = '2017-01-01'
end = '2021-12-31'
timeseries_df = messari.get_metric_timeseries(asset_slugs=assets, asset_metric=metrics, start=start, end=end, interval='1d')
timeseries_df.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/messari.csv', index = True)
