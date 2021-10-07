import numpy as np
import pandas as pd
from datetime import date
import datetime
#----------------------------

def construct_sentiment_feature_set(twitter_df, feature_df, ticker_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """

    if ticker_str == "TSLA":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_TSLA.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_TSLA.csv')

    if ticker_str ==  "GM":
        finance_short_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_short_GM.csv')
        finance_long_df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/finance_data_extended_GM.csv')




    feature_df = number_of_tweets(twitter_df, feature_df)

    """
    #get daily average score
    feature_df = daily_average_sentiment_basic(twitter_df, feature_df)
    feature_df = daily_average_sentiment(twitter_df, feature_df, "Stanford_sentiment")
    feature_df = daily_average_sentiment(twitter_df, feature_df, "TextBlob_sentiment")
    feature_df = daily_average_sentiment(twitter_df, feature_df, "Flair_sentiment")
    #get sentiment volatility
    feature_df = sentiment_volatility(twitter_df, feature_df)
    #get sentiment momentum
    feature_df = sentiment_momentum(twitter_df, feature_df, 5)
    """

    #add add_financials
    add_financials(finance_short_df,feature_df)
    #get same day's return
    same_day_return(finance_short_df, feature_df)
    #get previous day's return
    previous_day_return(finance_short_df, feature_df)
    #get daily volume
    volume(finance_short_df, feature_df)
    #get price momentum
    price_momentum(finance_short_df, feature_df, 5)
    #get price volatility
    price_volatility(finance_long_df, feature_df)

    return feature_df
#----------------------------

def number_of_tweets(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    date_count= pd.DataFrame(data=twitter_df['date_short'].value_counts())
    date_count.index = pd.to_datetime(date_count.index)
    date_count = date_count.rename(columns={'date_short':'number_of_tweets'})
    date_count['date'] = date_count.index

    feature_df = pd.merge(feature_df, date_count, how='left', on='date')
    print(date_count.number_of_tweets.sum())
    print(feature_df.number_of_tweets.sum())



    feature_df.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/feature_df_date_count.csv', index = False)
    date_count.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/date_count.csv', index = False)






    """


    unique_dates = twitter_df['date_short'].unique()
    date_count= pd.DataFrame(data=twitter_df['date_short'].value_counts())
    date_count.index = pd.to_datetime(date_count.index)
    for i, row in date_count.iterrows():
        feature_df.loc[feature_df['date'] == i, ['number of tweets']] = row['date_short']
    """
    return feature_df

#----------------------------

def daily_average_sentiment_basic(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby('date_short', as_index=False)['Stanford_sentiment'].mean()
    df.date_short = pd.to_datetime(df.date_short)
    for i, row in df.iterrows():
        feature_df.loc[feature_df['date'] == row['date_short'], ['daily average sentiment score']] = row['Stanford_sentiment']

    return feature_df

#----------------------------

def daily_average_sentiment(twitter_df, feature_df, sentiment_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby('date_short', as_index=False)[sentiment_str].mean()
    df.date_short = pd.to_datetime(df.date_short)
    row_name = "daily average "+sentiment_str+" score"
    for i, row in df.iterrows():
        feature_df.loc[feature_df['date'] == row['date_short'], [row_name]] = row[sentiment_str]

    return feature_df

#----------------------------

def sentiment_volatility(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    unique_dates = twitter_df['date_short'].unique()
    unique_dates = pd.DataFrame(data=unique_dates, columns=['date_short'])
    for i, row in unique_dates.iterrows():
        std = twitter_df.loc[twitter_df['date_short']==row['date_short'],'Stanford_sentiment'].std()
        volatility = std**.5
        feature_df.loc[feature_df['date'] == row['date_short'], ['sentiment volatility']] = volatility

    return feature_df

#----------------------------
def sentiment_momentum(twitter_df, feature_df, d):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """

    for i, row in feature_df.iterrows():
        t_now = row['date']
        t_before = row['date'] - pd.Timedelta(days=d)
        s_now = row['daily average sentiment score']
        s_before = feature_df.loc[feature_df['date'] == t_before,'daily average sentiment score'].max()
        if s_before == 0:
            s_before = float("NaN")
        p = (s_now / s_before)*100
        feature_df.loc[feature_df['date'] == row['date'], ['sentiment momentum']] = p

    return feature_df

#----------------------------

def sentiment_reversal(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    # TO DO

    return feature_df

#----------------------------
#---------------------
def add_financials(finance_df, feature_df):
    for i, row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], ['open']] = row['1. open']
        feature_df.loc[feature_df['date'] == row['date'], ['close']] = row['4. close']

    return feature_df

#---------------------
#----------------------------
def same_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    for i, row in finance_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        feature_df.loc[feature_df['date'] == row['date'], "same day return"]= rtn

    return feature_df

#----------------------------
def previous_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    finance_df['1. open'] = finance_df['1. open'].shift(periods=1)
    finance_df['4. close'] = finance_df['4. close'].shift(periods=1)
    for i, row in finance_df.iterrows():
        rtn = row['4. close']/row['1. open']-1
        feature_df.loc[feature_df['date'] == row['date'], "previous day's return"]= rtn

    return feature_df

#----------------------------
def volume(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed volume
    """
    #previous day return
    for i, row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], "volume"]= row['6. volume']

    return feature_df

#----------------------------
def price_momentum(finance_df, feature_df, d):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: feature_df, same shape as df but with the inputed features
    """
    df_shifted = finance_df.shift(periods=d)
    df = (finance_df['4. close']-df_shifted['4. close']).to_frame()
    df['date'] = finance_df['date']
    for i,row in df.iterrows():
        feature_df.loc[feature_df['date'] == row['date'], ['price momentum']] = row['4. close']

    return df

#----------------------------
def price_volatility(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    finance_df['time'] = pd.to_datetime(finance_df['time'])
    finance_df = finance_df.groupby([finance_df['time'].dt.date]).std()
    finance_df['close'] = finance_df['close']**.5
    for i,row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == pd.Timestamp(i), ['price volatility']] = row['close']

    return feature_df
