import numpy as np
import pandas as pd
from _4.1_Feature_functions import number_of_tweets, daily_average_sentiment, sentiment_volatility, sentiment_momentum

ticker_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv')
product_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_TSLA.csv')
ticker_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
product_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
tweet_list=[ticker_set_TSLA, product_set_TSLA, ticker_set_GM, product_set_GM]

#initiate the feature dataframes where we can input the different features
col =["date","number of tweets","daily average sentiment score","sentiment volatility","sentiment momentum","reversal","previous day's return", "volume", "price momentum", "price volatility"]
Feature_set = pd.DataFrame(columns=col)
unique_dates = tweet_list[0]['date_short'].unique()
df = pd.DataFrame(data=unique_dates, columns=['date'])
df['date'] = pd.to_datetime(df['date'])
Feature_set = Feature_set.append(df,ignore_index=True )

Feature_set_Ticker_TSLA = Feature_set
Feature_set_Product_TSLA = Feature_set
Feature_set_Ticker_GM = Feature_set
Feature_set_Product_GM = Feature_set
feature_list =[Feature_set_Ticker_TSLA,Feature_set_Product_TSLA,Feature_set_Ticker_GM,Feature_set_Product_GM]

for t in tweet_list:
    t['date_short']=pd.to_datetime(t['date_short'])
    ticker_tweets = t
    for i in feature_list:
        #get number of tweets
        i = number_of_tweets(ticker_tweets, i)
        #get daily average score
        i = daily_average_sentiment(ticker_tweets, i)
        #get sentiment volatility
        i = sentiment_volatility(ticker_tweets, i)
        #get sentiment momentum
        i = sentiment_momentum(ticker_tweets, i, 5)
        #get sentiment reversal
        # TODO
        #get previous day's return
        previous_day_return(TSLA,i)
        #get daily volume
        volume(TSLA, i)
        #get price momentum
        price_momentum(TSLA, i, 5)
        #get price volatility
        price_volatility("TSLA", start, end, i)

feature_list[0].to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_TSLA.csv', index = False)
feature_list[1].to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Product_TSLA.csv', index = False)
feature_list[2].to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_GM.csv', index = False)
feature_list[3].to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Product_GM.csv', index = False)
