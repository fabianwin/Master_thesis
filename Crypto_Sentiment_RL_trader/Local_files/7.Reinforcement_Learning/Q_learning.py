import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C


"""
#loading our dataset
df = pd.read_csv('/finance_data_short_TSLA.csv')
#converting Date Column to DateTime Type
df['date'] = pd.to_datetime(df['date'])
df = df.drop(['5. adjusted close', '7. dividend amount','8. split coefficient'], axis=1)
print(df.dtypes)
#setting the column as index
df.set_index('date', inplace=True)
print(df)
"""

coin = "BTC"
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
my_file = 'complete_feature_set_'+coin+".csv"
date_cols = ["date"]
data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
print(data_df.head())
