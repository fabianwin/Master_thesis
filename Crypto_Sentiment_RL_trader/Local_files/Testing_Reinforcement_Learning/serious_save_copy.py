import pandas as pd
import numpy as np
import random
from collections import deque
from utils import TradingGraph, Write_to_file

class CustomEnv:
    # A custom Bitcoin trading environment, render range is the amount of dates shown after this FIFO
    def __init__(self, df, initial_balance=1000, lookback_window_size=10, Render_range = 100):
        # Define action space and state size and other custom parameters
        self.df = df.reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range # render range in visualization

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        # 10 = size of orders information (balance, net worth,...) and market inof (OHCLV)
        self.state_size = (self.lookback_window_size, 10)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(Render_range=self.Render_range) # init visualization
        self.trades = deque(maxlen=self.Render_range) # limited orders memory for visualization

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0

        # we don't want to train our agent on the same set. We hence choose a random part from it in the size of the training_batch_size
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Price (Open)'],
                                        self.df.loc[current_step, 'Price (High)'],
                                        self.df.loc[current_step, 'Price (Low)'],
                                        self.df.loc[current_step, 'Price (Close)'],
                                        self.df.loc[current_step, 'Real Volume']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Price (Open)'],
                                    self.df.loc[self.current_step, 'Price (High)'],
                                    self.df.loc[self.current_step, 'Price (Low)'],
                                    self.df.loc[self.current_step, 'Price (Close)'],
                                    self.df.loc[self.current_step, 'Real Volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close because we can't know which price it will actually cost
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Price (Open)'],
            self.df.loc[self.current_step, 'Price (Close)'])
        Date = self.df.loc[self.current_step, 'date'] # for visualization
        High = self.df.loc[self.current_step, 'Price (High)'] # for visualization
        Low = self.df.loc[self.current_step, 'Price (Low)'] # for visualization

        if action == 0: # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_bought, 'type': "buy"})

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        #Write_to_file(Date, self.orders_history[-1])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    # render environment
    def render(self, visualize = False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            Date = self.df.loc[self.current_step, 'date']
            Open = self.df.loc[self.current_step, 'Price (Open)']
            Close = self.df.loc[self.current_step, 'Price (Close)']
            High = self.df.loc[self.current_step, 'Price (High)']
            Low = self.df.loc[self.current_step, 'Price (Low)']
            Volume = self.df.loc[self.current_step, 'Real Volume']

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)


def Random_games(env, visualize, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        while True:
            env.render(visualize)

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)

####MAIN###
df = pd.read_csv(r'/Users/fabianwinkelmann/github/Master_Thesis/Crypto_Sentiment_RL_trader/Local_files/7.Reinforcement_Learning/part1/pricedata.csv')
df = df.sort_values('Date')
#print(df.info(verbose=True))

df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')

lookback_window_size = 10
train_df = df[:-280-lookback_window_size]
test_df = df[-280-lookback_window_size:]

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, visualize=True, train_episodes = 1, training_batch_size=360)