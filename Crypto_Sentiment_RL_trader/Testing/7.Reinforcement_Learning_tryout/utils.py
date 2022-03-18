#================================================================
#
#   File name   : utils.py
#   Author      : PyLessons
#   Created date: 2021-02-25
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : additional functions
#
#================================================================
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np

def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
    for i in net_worth:
        Date += " {}".format(i)
    #print(Date)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()

def display_frames_as_gif(frames, episode):
    import pylab
    from matplotlib import animation
    try:
        pylab.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = pylab.imshow(frames[0])
        pylab.axis('off')
        pylab.subplots_adjust(left=0, right=1, top=1, bottom=0)
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(pylab.gcf(), animate, frames = len(frames), interval=33)
        anim.save(str(episode)+'_gameplay.gif')
    except:
        pylab.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
        patch = pylab.imshow(frames[0])
        pylab.axis('off')
        pylab.subplots_adjust(left=0, right=1, top=1, bottom=0)
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(pylab.gcf(), animate, frames = len(frames), interval=33)
        anim.save(str(episode)+'_gameplay.gif', writer=animation.PillowWriter(fps=10))

class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, Render_range, Show_reward=False, Show_indicators=False):
        self.Volume = deque(maxlen=Render_range)
        self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range
        self.Show_reward = Show_reward
        self.Show_indicators = Show_indicators

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

        # Add paddings to make graph easier to view
        #plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        # define if show indicators
        if self.Show_indicators:
            self.Create_indicators_lists()

    def Create_indicators_lists(self):
        # Create a new axis for indicatorswhich shares its x-axis with volume
        #self.ax4 = self.ax2.twinx()

        self.sma7 = deque(maxlen=self.Render_range)
        self.sma25 = deque(maxlen=self.Render_range)
        self.sma99 = deque(maxlen=self.Render_range)

        self.bb_bbm = deque(maxlen=self.Render_range)
        self.bb_bbh = deque(maxlen=self.Render_range)
        self.bb_bbl = deque(maxlen=self.Render_range)

        self.psar = deque(maxlen=self.Render_range)

        self.MACD = deque(maxlen=self.Render_range)
        self.RSI = deque(maxlen=self.Render_range)


    def Plot_indicators(self, df, Date_Render_range):
        self.sma7.append(df["sma7"])
        self.sma25.append(df["sma25"])
        self.sma99.append(df["sma99"])

        self.bb_bbm.append(df["bb_bbm"])
        self.bb_bbh.append(df["bb_bbh"])
        self.bb_bbl.append(df["bb_bbl"])

        self.psar.append(df["psar"])

        self.MACD.append(df["MACD"])
        self.RSI.append(df["RSI"])

        # Add Simple Moving Average
        self.ax1.plot(Date_Render_range, self.sma7,'-')
        self.ax1.plot(Date_Render_range, self.sma25,'-')
        self.ax1.plot(Date_Render_range, self.sma99,'-')

        # Add Bollinger Bands
        self.ax1.plot(Date_Render_range, self.bb_bbm,'-')
        self.ax1.plot(Date_Render_range, self.bb_bbh,'-')
        self.ax1.plot(Date_Render_range, self.bb_bbl,'-')

        # Add Parabolic Stop and Reverse
        self.ax1.plot(Date_Render_range, self.psar,'.')

        self.ax4.clear()
        # # Add Moving Average Convergence Divergence
        self.ax4.plot(Date_Render_range, self.MACD,'r-')

        # # Add Relative Strength Index
        self.ax4.plot(Date_Render_range, self.RSI,'g-')


    # Render the environment to the screen
    #def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
    def render(self, df, net_worth, trades):
        Date = df["date"]
        Open = df["Price (Open)"]
        High = df["Price (High)"]
        Low = df["Price (Low)"]
        Close = df["Price (Close)"]
        Volume = df["Real Volume"]
        # append volume and net_worth to deque list
        self.Volume.append(Volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])

        # Clear the frame rendered last step
        self.ax1.clear()
        #candlestick_ohlc(self.ax1, self.render_data, width=0.8, colorup='green', colordown='red', alpha=0.8)

        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, self.Volume, 0)

        if self.Show_indicators:
            self.Plot_indicators(df, Date_Render_range)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="orange")

        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:,1:])
        maximum = np.max(np.array(self.render_data)[:,1:])
        RANGE = maximum - minimum

        """
        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low'] - RANGE*0.02
                    ycoords = trade['Low'] - RANGE*0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High'] + RANGE*0.02
                    ycoords = trade['High'] + RANGE*0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")

                if self.Show_reward:
                    try:
                        self.ax1.annotate('{0:.2f}'.format(trade['Reward']), (trade_date-0.02, high_low), xytext=(trade_date-0.02, ycoords),
                                                   bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
                    except:
                        pass
        """

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')


        ######
        df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv', index_col=1, parse_dates=True)
        df = df[-151:]
        fund = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data/BITX_index_fund.csv',index_col=0, parse_dates=True)
        fund = fund[-104:]
        new_df = pd.merge(df, fund, how='left',  left_on="date", right_index=True)
        new_df.fillna(method='ffill',inplace=True)

        color = 'tab:green'
        #self.ax1.set_xlabel('Date')
        #self.ax1.set_ylabel('BTC', color=color)
        self.ax1.plot(new_df.index, new_df['Price (Close)'], color=color)
        self.ax1.tick_params(axis='y', labelcolor=color)
        #self.ax2 = ax1.twinx()
        color = 'tab:purple'
        #self.ax1.set_ylabel("BITW",color=color)
        self.ax1.plot(new_df.index, new_df["Close"], color=color)
        self.ax1.tick_params(axis='y', labelcolor=color)
        self.ax3.set_yticks((1000, 1200, 1400, 1600, 1800, 2000, 2200, 2300))


        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        self.fig.suptitle('Random action agent', fontsize=16)
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/github/Master_thesis/Crypto_Sentiment_RL_trader/Local_files/7.Reinforcement_Learning')
        my_file = 'good_agent.png'
        plt.savefig(os.path.join(my_path, my_file), transparent=True)
        #plt.savefig(os.path.join(my_path, my_file))
        #Show the graph without blocking the rest of the program
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)




def Plot_OHCL(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["date"] = pd.to_datetime(df.Date)
    df["date"] = df["date"].apply(mpl_dates.date2num)

    df = df[['date', 'Price (Open)', 'Price (High)', 'Price (Low)', 'Price (Close)', 'Real Volume']]

    # We are using the style ‘ggplot’
    plt.style.use('ggplot')

    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8))

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)


    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))# %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()

def Normalizing(df_original):
    df = df_original.copy()
    column_names = df.columns.tolist()
    for column in column_names[1:]:
        # Logging and Differencing
        test = np.log(df[column]) - np.log(df[column].shift(1))
        if test[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        # Min Max Scaler implemented
        Min = df[column].min()
        Max = df[column].max()
        df[column] = (df[column] - Min) / (Max - Min)

    return df

if __name__ == "__main__":
    # testing normalization technieques
    df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading/complete_feature_set_BTC.csv')
    feature_list = ["date","Price (Open)","Price (High)","Price (Low)","Price (Close)","Real Volume","ticker_number_of_tweets", "ticker_finiteautomata_sentiment", "ticker_finiteautomata_sentiment_expectation_value_volatility", "ticker_average_number_of_followers","Momentum_14_ticker_finiteautomata_sentiment","MOM_14","Volatility","RSI_14"]
    df = df.loc[:,feature_list]

    #df["Close"] = df["Close"] - df["Close"].shift(1)
    df["Price (Close)"] = np.log(df["Price (Close)"]) - np.log(df["Price (Close)"].shift(1))

    Min = df["Price (Close)"].min()
    Max = df["Price (Close)"].max()
    df["Price (Close)"] = (df["Price (Close)"] - Min) / (Max - Min)

    fig = plt.figure(figsize=(16,8))
    plt.plot(df["Price (Close)"],'-')
    ax=plt.gca()
    ax.grid(True)
    fig.tight_layout()
    #plt.show()
