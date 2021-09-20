import numpy as np
import pandas as pd
#----------------------------

def number_of_tweets(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    unique_dates = twitter_df['date_short'].unique()
    date_count= pd.DataFrame(data=twitter_df['date_short'].value_counts())
    for i, row in date_count.iterrows():
        feature_df.loc[feature_df['date'] == i, ['number of tweets']] = row['date_short']

    return feature_df

#----------------------------

def daily_average_sentiment(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    unique_dates = twitter_df['date_short'].unique()
    unique_dates = pd.DataFrame(data=unique_dates, columns=['date_short'])
    unique_dates['date_short'] = pd.to_datetime(unique_dates['date_short'])
    for i, row in unique_dates.iterrows():
        score = twitter_df.loc[twitter_df['date_short']==row['date_short'],'Stanford_sentiment'].mean()
        feature_df.loc[feature_df['date'] == row['date_short'], ['daily average sentiment score']] = score

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
def previous_day_return(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    #previous day return
    for i, row in finance_df.iterrows():
        rtn = row['Open']/row['Close']-1
        feature_df.loc[feature_df['date'] == i, "previous day's return"]= rtn

    return feature_df

#----------------------------
def volume(finance_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed volume
    """
    #previous day return
    for i, row in finance_df.iterrows():
        feature_df.loc[feature_df['date'] == i, "volume"]= row['Volume']

    return feature_df

#----------------------------
def price_momentum(finance_df, feature_df, d):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: feature_df, same shape as df but with the inputed features
    """
    df_shifted = finance_df.shift(periods=d)
    df = (finance_df['Close']-df_shifted['Close']).to_frame()
    for i,row in df.iterrows():
        feature_df.loc[feature_df['date'] == i, ['price momentum']] = row['Close']

    return feature_df

#----------------------------
def price_volatility(ticker,start, end, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = yf.download("TSLA",start,end, interval="5m")
    df.index = pd.to_datetime(df.index)
    df = df.groupby([df.index.date]).std()
    df['Close'] = df['Close']**.5

    #feature_df['date'] = feature_df['date'].dt.time
    for i,row in df.iterrows():
        feature_df.loc[feature_df['date'] == pd.Timestamp(i), ['price volatility']] = row['Close']

    return feature_df
