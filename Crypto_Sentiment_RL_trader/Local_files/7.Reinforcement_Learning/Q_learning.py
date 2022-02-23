import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines.common.vec_env import DummyVecEnv

#load the data
coin = "BTC"
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
my_file = 'complete_feature_set_'+coin+".csv"
date_cols = ["date"]
data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
data_df.rename(columns={"Price (Close)": "Close"}, inplace = True)

#print(data_df["MOM_14"])

#add custom signals
def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close','ROC_2', 'ticker_number_of_tweets', 'MOM_14']].to_numpy()[start:end]
    return prices, signal_features

class MyForexEnv(StocksEnv):
    _process_data = add_signals

env = MyForexEnv(df=data_df, window_size=10, frame_bound=(10, len(data_df)))

#setting up our environment for training
env_maker = lambda: env
env = DummyVecEnv([env_maker])
#Applying the Trading RL Algorithm
model = A2C('MlpLstmPolicy', env, verbose=1)
#setting the learning timesteps
model.learn(total_timesteps=1000)


"""
print("prices", env.prices)
print()
print("signals shape", env.signal_features.shape)
print("signals", env.signal_features)
print()
print( "Action space", env.action_space)
print()
print( "Observation space", env.observation_space)


#running the test environment with random actions
state = env.reset()
while True:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done:
        print("info", info)
        break
#plot the results
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()
"""
