import numpy as np
import pandas as pd
from Featurefunctions import number_of_tweets, daily_average_sentiment, sentiment_volatility, sentiment_momentum


ticker_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_1month.csv')
#Feature_set_Ticker = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_1month.csv')

print(ticker_tweets.head())



#type conversion for the ticker_tweet df
ticker_tweets['date_short']=pd.to_datetime(ticker_tweets['date_short'])

#initiate the feature dataframes where we can input the different features
col =["date","number of tweets","daily average sentiment score","sentiment volatility","sentiment momentum","reversal","previous day's return", "volume", "price momentum", "price volatility"]
Feature_set = pd.DataFrame(columns=col)
unique_dates = ticker_tweets['date_short'].unique()
df = pd.DataFrame(data=unique_dates, columns=['date'])
df['date'] = pd.to_datetime(df['date'])

Feature_set = Feature_set.append(df,ignore_index=True )

Feature_set_Ticker = Feature_set
Feature_set_Product = Feature_set

#get number of tweets
Feature_set_Ticker = number_of_tweets(ticker_tweets, Feature_set_Ticker)
#get daily average score
Feature_set_Ticker = daily_average_sentiment(ticker_tweets, Feature_set_Ticker)
#get sentiment volatility
Feature_set_Ticker = sentiment_volatility(ticker_tweets, Feature_set_Ticker)
#get sentiment momentum
Feature_set_Ticker = sentiment_momentum(ticker_tweets, Feature_set_Ticker, 5)
#get sentiment reversal
# TODO

print("Sentiment features are calculated")
