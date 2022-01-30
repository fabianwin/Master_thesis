import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from lppls import LPPLS
from datetime import datetime, date, time, timezone
import pytz
from sklearn import preprocessing

def get_lppls_graphs(symbol):
    # read example dataset into df
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
    my_file = 'finance_data_'+symbol+".csv"
    data = pd.read_csv(os.path.join(my_path, my_file))
    data.dropna(axis=0, how='any',subset=['Price (Close)'], inplace=True)

    # convert time to ordinal
    time = [pd.Timestamp.toordinal(dt.strptime(t1, '%d.%m.%y')) for t1 in data['Date']]

    # create list of observation data
    price = np.log(data['Price (Close)'].values)

    # create observations array (expected format for LPPLS observations)
    observations = np.array([time, price])

    # set the max number for searches to perform before giving-up
    # the literature suggests 25
    MAX_SEARCHES = 25

    # instantiate a new LPPLS model with the Nasdaq Dot-com bubble dataset
    lppls_model = LPPLS(observations=observations)

    # fit the model to the data and get back the params
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)
    lppls_model.plot_fit(symbol)

    if __name__ == '__main__':
        print('compute LPPLS conf scores fresh')
        # compute the confidence indicator
        res = lppls_model.mp_compute_nested_fits(
            workers=CPU_CORES,
            window_size=126*3,
            smallest_window_size=21,
            outer_increment=1,
            inner_increment=5,
            max_searches=25,
            # filter_conditions_config={} # not implemented in 0.6.x
        )
        res_df = lppls_model.compute_indicators(res)
        res_df['time'] = [pd.Timestamp.fromordinal(int(t1)) for t1 in res_df['time']]
        res_df.set_index('time', inplace=True)
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
        my_file = 'LPPLS_CONF_CSV_'+symbol+".csv"
        res_df.to_csv(os.path.join(my_path, my_file))

        lppls_model.plot_confidence_indicators(res, symbol)
        # should give a plot like the following...
#----------------------------
# Complete Functions
#---------------------
def construct_sentiment_feature_set(twitter_df, symbol, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    col =["same_hour_return","sentiment volatility", "sentiment momentum"]
    df = pd.DataFrame({'date': pd.date_range(start="2017-01-01",end="2021-12-31")})
    Feature_set = df.reindex(columns = df.columns.tolist())

    #get number of tweets
    Feature_set = number_of_tweets(twitter_df, Feature_set, set_description)
    #get average number of tweet likes
    Feature_set = likes_of_tweets(twitter_df, Feature_set, set_description)
    #get number of tweets
    #Feature_set = number_of_tweets(twitter_df, Feature_set, set_description)
    #get number of tweets
    #Feature_set = number_of_tweets(twitter_df, Feature_set, set_description)

    #get daily average score
    #Feature_set = daily_average_sentiment(twitter_df, Feature_set, "Stanford Sentiment")
    #Feature_set = daily_average_sentiment(twitter_df, Feature_set, "TextBlob Sentiment")
    #Feature_set = daily_average_sentiment(twitter_df, Feature_set, "Flair Sentiment")
    #Feature_set = normalized_average_sentiment(Feature_set)
    """
    #get sentiment volatility
    feature_df = sentiment_volatility(twitter_df, feature_df)
    #get sentiment momentum
    feature_df = sentiment_momentum(twitter_df, feature_df, 8)
    #add add_financials (open, close, volume)
    feature_df = add_financials(finance_long_df,feature_df)
    #get same day's return
    feature_df = same_hour_return(feature_df)
    #get same day's return
    feature_df = next_hour_return (feature_df)
    #get previous day's return
    feature_df = previous_hour_return(feature_df)
    #get price momentum
    feature_df = price_momentum(feature_df, 15)
    #get price volatility
    feature_df = price_volatility(feature_df)


    pd.set_option('display.max_columns', None)
    print(feature_df)
    pd.reset_option('display.max_rows')
    """
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering')
    my_file = 'feature_set_'+symbol+".csv"
    Feature_set.to_csv(os.path.join(my_path, my_file))

    return
#Sentiment functions
#----------------------------
def number_of_tweets(twitter_df, feature_df, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    date_count= pd.DataFrame(data=twitter_df['date_short'].value_counts())
    date_count.index = pd.to_datetime(date_count.index)
    new_column_name = set_description+"_number_of_tweets"
    date_count = date_count.rename(columns={'date_short':new_column_name})
    date_count['date'] = date_count.index
    feature_df = pd.merge(feature_df, date_count, how='left', on='date')

    return feature_df
#----------------------------
def likes_of_tweets(twitter_df, feature_df, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby(pd.Grouper(key='date_short',freq='D')).sum()
    df.index = pd.to_datetime(df.index)
    feature_df = pd.merge(feature_df, df.likes, how='left',  left_on="date", right_on="date_short")
    tweet_number_column_name = set_description+"_number_of_tweets"
    new_column_name = set_description+"_average_number_of_likes"
    feature_df[new_column_name] = feature_df.likes / feature_df[tweet_number_column_name]
    feature_df.drop(['likes'], axis=1, inplace=True)

    return feature_df
#----------------------------
def hourly_average_sentiment(twitter_df, feature_df, sentiment_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby('date_medium', as_index=False)[sentiment_str].mean()
    df.date_medium = pd.to_datetime(df.date_medium)
    feature_df = pd.merge(feature_df, df, how='left', left_on='date', right_on='date_medium')
    feature_df= feature_df.drop(['date_medium'], axis=1)

    return feature_df
#----------------------------
def normalized_average_sentiment(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = feature_df.iloc[:,[7,8,9]]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df)
    normalized_df = pd.DataFrame(x_scaled)
    normalized_df['normalized average sentiment'] = normalized_df.iloc[:, 0:2].mean(axis=1)
    normalized_df['date'] = feature_df['date']
    feature_df = pd.merge(feature_df, normalized_df[['date','normalized average sentiment']], how='left', on='date')

    return feature_df
#----------------------------
def sentiment_volatility(twitter_df, feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    unique_dates = twitter_df['date_medium'].unique()
    unique_dates = pd.DataFrame(data=unique_dates, columns=['date_medium'])
    for i, row in unique_dates.iterrows():
        std = twitter_df.loc[twitter_df['date_medium']==row['date_medium'],'Stanford Sentiment'].std()
        volatility = std**.5
        feature_df.loc[feature_df['date'] == row['date_medium'], ['sentiment volatility']] = volatility

    return feature_df
#----------------------------
def sentiment_momentum(twitter_df, feature_df, h):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    for i, row in feature_df.iterrows():
        t_now = row['date']
        t_before = row['date'] - pd.Timedelta(hours=h)
        s_now = row['Stanford Sentiment']
        s_before = feature_df.loc[feature_df['date'] == t_before,'Stanford Sentiment'].max()
        if s_before == 0:
            p = float("NooooN")
        else:
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
# Finance Functions
#---------------------
def add_financials(finance_df, feature_df):
    finance_df.time = pd.to_datetime(finance_df.time, utc=True)
    finance_df = finance_df.rename(columns={'time':'date'})
    finance_df = finance_df.drop(['high','low'], axis=1)
    feature_df = pd.merge(feature_df, finance_df, how='left', on='date')
    #date_count.index = pd.to_datetime(date_count.index)
    return feature_df
#----------------------------
def same_hour_return(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.same_hour_return = feature_df.close/feature_df.open -1

    return feature_df
#----------------------------
def next_hour_return (feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.next_hour_return = feature_df.same_hour_return.shift(periods=-1)

    return feature_df
#----------------------------
def previous_hour_return(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: feature_df, same shape as df but with the inputed previous day's return
    """
    feature_df.previous_hour_return = feature_df.same_hour_return.shift(periods=1)

    return feature_df
#----------------------------
def price_momentum(feature_df, h):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: feature_df, same shape as df but with the inputed features
    """

    df = pd.DataFrame([feature_df.date, feature_df.close]).transpose()
    df = df.dropna(axis=0, how='any')
    df['shifted_close'] = df.close.shift(periods=h)
    df['momentum'] = df.close - df.shifted_close
    df = df.drop(['close','shifted_close'], axis=1)
    feature_df = pd.merge(feature_df, df, how='left', on='date')

    return feature_df
#----------------------------
def price_volatility(feature_df):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = feature_df
    #calculate log returns
    df['log returns'] = np.log(feature_df['close']/feature_df['close'].shift())
    df = df.groupby(pd.Grouper(key='date',freq='D')).std()
    #calcualte volatility based on 15 trading hours per day. Every day will have the same value, could be improved with minute data.
    df['volatility'] = df['log returns']*15**.5

    #merge the dataframes together and drop unnecessairy columns
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index
    df = df.iloc[:,[13]]
    feature_df = pd.merge(feature_df, df, left_on=[feature_df['date'].dt.year, feature_df['date'].dt.month, feature_df['date'].dt.day], right_on=[df.index.year, df.index.month,df.index.day], how='left')
    feature_df = feature_df.drop(['key_0', 'key_1', 'key_2'], axis=1)

    return feature_df
#---------------------
