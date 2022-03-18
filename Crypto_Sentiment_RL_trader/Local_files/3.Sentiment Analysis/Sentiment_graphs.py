import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from warnings import filterwarnings
import pingouin as pg
filterwarnings('ignore')
sns.set()

#----------------------------
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
#----------------------------
def plot_sentiment_price_correlation(feature_df, symbol, set_description):
    sentiment = set_description+"_finiteautomata_sentiment"
    price = "same_day_return"
    g = sns.jointplot(sentiment, price, data=feature_df, kind='reg',scatter_kws={'s': 1})
    df = feature_df[[sentiment, price]]
    df.dropna(inplace=True)
    r, p = stats.pearsonr(df[sentiment], df[price])
    g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                    xy=(0.08, 0.88), xycoords='axes fraction',
                    ha='left', va='bottom',
                    bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})

    g.ax_joint.set_ylabel('Same Day Return')
    g.ax_joint.set_xlabel('Daily Average BERTweet Sentiment')
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/Graphs') # Figures out the absolute path for you in case your working directory moves around.
    my_file = set_description+"_sentiment_return_correlation_"+symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))

    plt.show()
#----------------------------
def plot_sentiment_price_graph(df, symbol, set_description):
    df.index = df.date
    sentiment = set_description+"_finiteautomata_sentiment"

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel('Daily Average BERTweet Sentiment', color=color)
    ax1.plot(df.index, df[sentiment], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    y_label = coin + " Value"
    ax2.set_ylabel(y_label,color=color)
    ax2.plot(df.index, df["Price (Close)"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/Graphs')
    my_file = set_description+"_sentiment_value_graph_"+symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))
    plt.show()

#----------------------------
#Main
coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
sets=["ticker", "news"]

for coin in coins:
    for set in sets:
        #read data
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
        my_file = 'complete_feature_set_'+coin+".csv"
        date_cols = ["date"]
        data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
        plot_sentiment_price_graph(data_df,coin, set)
        #plot_sentiment_price_correlation(data_df,coin, set)
