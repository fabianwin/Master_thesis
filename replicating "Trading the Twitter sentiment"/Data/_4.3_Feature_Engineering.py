import yfinance as yf
import pandas as pd
import math

start = "2021-04-01"
end = '2021-08-30'

TSLA = yf.download('TSLA',start,end)
TSLA.index = pd.to_datetime(TSLA.index)

TSLA
