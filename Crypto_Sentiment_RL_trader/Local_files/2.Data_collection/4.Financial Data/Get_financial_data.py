import pandas as pd

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Binance_BTCUSDT_1h.csv', skiprows=1)
df.set_index('date', inplace=True)
print(df.info())

print("   ")
print("   ")

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Binance_ETHUSDT_1h.csv', skiprows=1)
df.set_index('date', inplace=True)
print(df.info())

print("   ")
print("   ")

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Binance_BNBUSDT_1h.csv', skiprows=1)
df.set_index('date', inplace=True)
print(df.info())

print("   ")
print("   ")

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Binance_ADAUSDT_1h.csv', skiprows=1)
df.set_index('date', inplace=True)
print(df.info())

print("   ")
print("   ")

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Binance_XRPUSDT_1h.csv', skiprows=1)
df.set_index('date', inplace=True)
print(df.info())
