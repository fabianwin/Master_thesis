import pandas as pd
import os

coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']


for coin in coins:
    #get coin data
    print(coin)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data_copy/Daily_Data')
    my_file = 'Coin_data_combined_'+coin+".csv"
    date_cols = ["Date"]
    coin_data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
    dates = pd.date_range(start="2017-01-01",end="2021-12-31")
    #exception for ADA
    if coin =='ADA':
        coin_data_df['Date'] = pd.date_range(start="2017-09-23",end="2021-12-31")
    #exception for BNB
    elif coin == 'BNB' :
        dates = pd.date_range(start="2017-07-05",end="2021-12-31")
        coin_data_df['Date'] = dates.sort_values(ascending=False)
        coin_data_df = coin_data_df.sort_values(by='Date',ascending=True)
    elif coin == 'XRP' :
        dates = pd.date_range(start="2017-01-01",end="2021-12-31")
        coin_data_df['Date'] = dates.sort_values(ascending=False)
        coin_data_df = coin_data_df.sort_values(by='Date',ascending=True)
    elif coin == 'ETH' :
        dates = pd.date_range(start="2017-01-01",end="2021-12-31")
        coin_data_df['Date'] = dates.sort_values(ascending=False)
        coin_data_df = coin_data_df.sort_values(by='Date',ascending=True)
    else:
        coin_data_df['Date'] = pd.date_range(start="2017-01-01",end="2021-12-31")
    print(coin_data_df.head(5))
    print(coin_data_df.tail(5))
    print(coin_data_df.Date)
    print(coin," finance feature completed")
    print("--------------------------------")
    print( )

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data_copy/Daily_Data')
    my_file = 'Coin_data_combined_'+coin+".csv"
    coin_data_df.to_csv(os.path.join(my_path, my_file))
