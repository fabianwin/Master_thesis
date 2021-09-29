import pandas as pd
import numpy as np
from Feature_functions import construct_sentiment_feature_set, number_of_tweets, daily_average_sentiment, sentiment_volatility, sentiment_momentum

ticker_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv')
product_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_TSLA.csv')
ticker_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
product_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_GM.csv')
finance_data_short_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_TSLA.csv')
finance_data_extended_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_TSLA.csv')
finance_data_short_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_GM.csv')
finance_data_extended_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_GM.csv')


#initiate the feature dataframes where we can input the different features
col =["date","number of tweets","daily average sentiment score","sentiment volatility","sentiment momentum","reversal","previous day's return", "volume", "price momentum", "price volatility"]
Feature_set = pd.DataFrame(columns=col)
unique_dates = ticker_set_TSLA['date_short'].unique()
df = pd.DataFrame(data=unique_dates, columns=['date'])
df['date'] = pd.to_datetime(df['date'])
Feature_set = Feature_set.append(df,ignore_index=True)

#construct the feature sets and save them
Feature_set_Ticker_TSLA = construct_sentiment_feature_set(ticker_set_TSLA, Feature_set, finance_data_short_TSLA, finance_data_extended_TSLA)
Feature_set_Ticker_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_TSLA.csv', index = False)
print(Feature_set_Ticker_TSLA["price volatility"])

'''
Feature_set_Product_TSLA = construct_sentiment_feature_set(product_set_TSLA, Feature_set, finance_data_short_TSLA, finance_data_extended_TSLA)
Feature_set_Product_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Product_TSLA.csv', index = False)
'''

Feature_set_Ticker_GM = construct_sentiment_feature_set(ticker_set_GM, Feature_set, finance_data_short_GM, finance_data_extended_GM)
Feature_set_Ticker_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_GM.csv', index = False)
print(Feature_set_Ticker_GM)

#Feature_set_Product_GM = construct_sentiment_feature_set(product_set_GM, Feature_set)
#Feature_set_Product_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Product_GM.csv', index = False)
