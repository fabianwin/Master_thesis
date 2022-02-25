import pandas as pd
import os
import copy
import numpy as np
import random
from collections import deque
from matplotlib import pylab
from utils import TradingGraph, Write_to_file
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam, RMSprop
import gym as Env

class CryptoEnv(Env):
    # A custom Bitcoin trading environment, render range is the amount of dates shown, after this FIFO
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


        """
        # Neural Networks part bellow
        self.lr = 0.00001
        self.epochs = 1
        self.normalize_value = 100000
        self.optimizer = Adam

        # Create Actor-Critic network model
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        """


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
            self.episode_orders += 1

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'total': self.crypto_sold, 'type': "sell"})
            self.episode_orders += 1

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

        obs = self._next_observation() / self.normalize_value

        return self.state, obs, reward, done

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
        self.episode_orders = 0
        self.env_steps_size = env_steps_size

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
        return self.state

    # create tensorboard writer
    def create_writer(self):
        self.replay_count = 0
        self.writer = SummaryWriter(comment="Crypto_trader")

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


    # render environment
    def render(self, visualize = False):
        if visualize:
            Date = self.df.loc[self.current_step, 'date']
            Open = self.df.loc[self.current_step, 'Price (Open)']
            Close = self.df.loc[self.current_step, 'Price (Close)']
            High = self.df.loc[self.current_step, 'Price (High)']
            Low = self.df.loc[self.current_step, 'Price (Low)']
            Volume = self.df.loc[self.current_step, 'Real Volume']

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)

    # Generalized Advantage Estimation (GAE): Advantage can be defined as a way to measure how much better off we can be by taking a particular action when we are in a particular state
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        # Gamma = constant discount factor which discounts future discount_rewards
        # Lambda = smoothing parameter, advantges taking action now and in future
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        #discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute advantages
        #advantages = discounted_r - values
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        #uncomment if plotting gaes
        """
        pylab.plot(target,'-')
        pylab.plot(advantages,'.')
        ax=pylab.gca()
        ax.grid(True)
        pylab.show()
        """

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{name}_Critic.h5")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.Actor.Actor.load_weights(f"{name}_Actor.h5")
        self.Critic.Critic.load_weights(f"{name}_Critic.h5")
