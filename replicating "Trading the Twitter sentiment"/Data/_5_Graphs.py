import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import pandas as pd
import pingouin as pg
sns.set(style='white', font_scale=1.2)

#load df
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_TSLA.csv')

#check correlation find informaiton for the variables here https://raphaelvallat.com/correlation.html
pg.corr(x=Feature_set_Ticker_TSLA['daily average sentiment score'], y=Feature_set_Ticker_TSLA["previous day's return"])

#draw plot
g = sns.jointplot(data=Feature_set_Ticker_TSLA, x="daily average sentiment score", y="previous day's return")
#g.ax_joint.plot(145, 95, 'r = 0.008654, p < 0.88723', fontstyle='italic')
g.fig.text(145, 95, 'r = 0.008654, p < 0.88723', fontstyle='italic')
plt.show()


"""
#g = sns.JointGrid(data=Feature_set_Ticker_TSLA, x="daily avergae sentiment score", y="previous day's return", xlim=(140, 190), ylim=(40, 100), height=5)

"""
