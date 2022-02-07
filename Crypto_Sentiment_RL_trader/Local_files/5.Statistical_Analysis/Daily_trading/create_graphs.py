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

#Functions

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
def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right
#----------------------------
def corrplot(data, size_scale=1000, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )
#----------------------------
#check if necessairy
def get_coefficient_df(x_str,y_str, df):
    c = pg.corr(x=df[x_str],y=df[y_str])
    c['x'] = x_str
    c['y'] =y_str
    return c
#----------------------------
def plot_sentiment_price_correlation(feature_df, symbol, set_description):

    sentiment_1 = set_description+"_TextBlob_sentiment"
    sentiment_2 = set_description+"_Flair_sentiment"
    sentiment_3 = set_description+"_finiteautomata_sentiment"
    sentiment_4 = set_description+"_finiteautomata_sentiment_1"
    price_1 = "same_day_return"
    price_2 = "ROC_2"

    sentiment_array = [sentiment_1, sentiment_2, sentiment_3, sentiment_4]
    price_array = [price_1, price_2]
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2, 4)
    i=0
    for price in price_array:
        for sentiment in sentiment_array:
            g = sns.jointplot(sentiment, price, data=feature_df, kind='reg',scatter_kws={'s': 1})
            df = feature_df[[sentiment, price]]
            df.dropna(inplace=True)
            r, p = stats.pearsonr(df[sentiment], df[price])
            #print(pg.corr(x=df[sentiment], y=df[price])) from https://raphaelvallat.com/correlation.html
            g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                            xy=(0.45, 0.15), xycoords='axes fraction',
                            ha='left', va='bottom',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
            SeabornFig2Grid(g, fig, gs[i])
            i = i+1

    gs.tight_layout(fig)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/5.Statistical_Analysis/Daily_trading')
    my_file = set_description+"sentiment_price_correlation_"+symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))

    plt.show()
#----------------------------
def plot_twittermeta_price_correlation(feature_df, symbol, set_description):

    meta_1 = set_description+"_number_of_tweets"
    meta_2 = set_description+"_average_number_of_likes"
    meta_3 = set_description+"_average_number_of_retweets"
    meta_4 = set_description+"_average_number_of_followers"
    price_1 = "same_day_return"
    price_2 = "ROC_2"

    meta_array = [meta_1, meta_2, meta_3, meta_4]
    price_array = [price_1, price_2]
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2, 4)
    i=0
    for price in price_array:
        for meta in meta_array:
            print(feature_df[meta])
            g = sns.jointplot(meta, price, data=feature_df, kind='reg',scatter_kws={'s': 1})
            df = feature_df[[meta, price]]
            df.dropna(inplace=True)
            r, p = stats.pearsonr(df[meta], df[price])
            #print(pg.corr(x=df[sentiment], y=df[price])) from https://raphaelvallat.com/correlation.html
            g.ax_joint.annotate(f'$\\rho = {r:.3f}, p = {p:.3f}$',
                            xy=(0.45, 0.15), xycoords='axes fraction',
                            ha='left', va='bottom',
                            bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
            SeabornFig2Grid(g, fig, gs[i])
            i = i+1

    gs.tight_layout(fig)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/5.Statistical_Analysis/Daily_trading')
    my_file = set_description+"twittermeta_price_correlation_"+symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))

    plt.show()
#----------------------------
def plot_feature_set_correlation_matrix(feature_df,symbol):
    corr = feature_df.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    plt.figure(figsize=(50, 50))
    corrplot(corr)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/5.Statistical_Analysis/Daily_trading') # Figures out the absolute path for you in case your working directory moves around.
    my_file = "feature_set_correlation_matrix_"+symbol+'.png'
    plt.savefig(os.path.join(my_path, my_file))
#----------------------------
#Main
coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
sets=["ticker", "product"]

for coin in coins:
    for set in sets:
        #read data
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
        my_file = 'complete_feature_set_'+coin+".csv"
        my_scaled_file = 'scaled_complete_feature_set_'+coin+".csv"
        date_cols = ["date"]
        data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
        scaled_data_df = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)
        data_df.drop(['Unnamed: 0','Unnamed: 0_x','Unnamed: 0.1','Unnamed: 0.1.1','Date','Unnamed: 0_y','date'], axis=1, inplace=True)
        data_df.info(verbose=True)

        #plot_feature_set_correlation_matrix(data_df, coin)
        plot_sentiment_price_correlation(data_df,coin, set)
        plot_twittermeta_price_correlation(data_df,coin, set)
