import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from lppls import LPPLS
from datetime import datetime, date, time, timezone
import pytz
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#----------------------------
# GLobal Parameters
#---------------------
#sentiment columnvalues
sent_str_1 = "TextBlob_sentiment"
sent_str_2 = "Flair_sentiment"
sent_str_3 = "finiteautomata_sentiment"
#---------------------
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

    """
    ###keep temporaraily
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering')
    my_file = 'feature_set_'+symbol+".csv"
    date_cols = ["date"]
    Feature_set = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols)
    #######
    """

    #get number of tweets
    Feature_set = number_of_tweets(twitter_df, Feature_set, set_description)
    #get average number of tweet likes
    Feature_set = avg_likes_of_tweets(twitter_df, Feature_set, set_description)
    #get average number of tweet likes
    Feature_set = avg_retweets_of_tweets(twitter_df, Feature_set, set_description)
    #get average number of tweet followers
    Feature_set = avg_followers_of_tweets(twitter_df, Feature_set, set_description)
    #get daily average score
    Feature_set = daily_average_sentiment(twitter_df, Feature_set, "TextBlob_sentiment",set_description)
    Feature_set = daily_average_sentiment(twitter_df, Feature_set, "Flair_sentiment",set_description)
    Feature_set = daily_average_sentiment(twitter_df, Feature_set, "finiteautomata_sentiment",set_description)
    Feature_set = average_sentiment(Feature_set,set_description)
    #get sentiment volatility
    Feature_set = sentiment_volatility(twitter_df, Feature_set,set_description)
    #get sentiment rate of change
    Feature_set = sentiment_ROC(twitter_df, Feature_set, 21,set_description)
    #get sentiment momentum
    Feature_set = sentiment_MOM(twitter_df, Feature_set, 21,set_description)
    #get sentiment relative strenght index
    Feature_set = sentiment_RSI(twitter_df, Feature_set, 21,set_description)

    """
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

    return Feature_set
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
    #scale the feature between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    newer_column_name = new_column_name+"_scaled"
    feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def avg_likes_of_tweets(twitter_df, feature_df, set_description):
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
    #scale the feature between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    newer_column_name = new_column_name+"_scaled"
    feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def avg_retweets_of_tweets(twitter_df, feature_df, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby(pd.Grouper(key='date_short',freq='D')).sum()
    df.index = pd.to_datetime(df.index)
    feature_df = pd.merge(feature_df, df.retweets, how='left',  left_on="date", right_on="date_short")
    tweet_number_column_name = set_description+"_number_of_tweets"
    new_column_name = set_description+"_average_number_of_retweets"
    feature_df[new_column_name] = feature_df.retweets / feature_df[tweet_number_column_name]
    feature_df.drop(['retweets'], axis=1, inplace=True)
    #scale the feature between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    newer_column_name = new_column_name+"_scaled"
    feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def avg_followers_of_tweets(twitter_df, feature_df, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    df = twitter_df.groupby(pd.Grouper(key='date_short',freq='D')).sum()
    df.index = pd.to_datetime(df.index)
    feature_df = pd.merge(feature_df, df['followers Count'], how='left',  left_on="date", right_on="date_short")
    tweet_number_column_name = set_description+"_number_of_tweets"
    new_column_name = set_description+"_average_number_of_followers"
    feature_df[new_column_name] = feature_df['followers Count']/ feature_df[tweet_number_column_name]
    feature_df.drop(['followers Count'], axis=1, inplace=True)
    #scale the feature between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    newer_column_name = new_column_name+"_scaled"
    feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def daily_average_sentiment(twitter_df, feature_df, sentiment_str, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    if sentiment_str=="finiteautomata_sentiment":
        twitter_df = get_expectation_value_column(twitter_df,sentiment_str, set_description)
        column_name_expectation = sentiment_str+"_expectation_value"
        df_sum = twitter_df.groupby(['date_short',sentiment_str], as_index=False)[column_name_expectation].sum()
        dff_sum = df_sum.groupby(['date_short'], as_index=False)[column_name_expectation].sum()
        unique_dates = df_sum['date_short'].unique()
        df = pd.DataFrame({'date_short': unique_dates})
        for index,row in df.iterrows():
            date = row['date_short']
            count_df = twitter_df.loc[(twitter_df['date_short'] == date) & ((twitter_df[sentiment_str] == "NEG")|(twitter_df[sentiment_str] == "POS"))]
            count = count_df[sentiment_str].count()
            expectation_df = dff_sum.loc[dff_sum['date_short'] == date]
            expectation_value = expectation_df.iloc[0][column_name_expectation]
            score = expectation_value/count
            ###alternative scoring mechanism wiht *_1 is divided by all entries to account for neutrals
            count_1_df = twitter_df.loc[(twitter_df['date_short'] == date)]
            count_1 = count_1_df[sentiment_str].count()
            score_1 = expectation_value/count_1
            new_column_name_1 = set_description+"_"+sentiment_str+"_1"
            ###
            new_column_name = set_description+"_"+sentiment_str
            df.at[index, new_column_name] =  score
            df.at[index, new_column_name_1] =  score_1

        feature_df = pd.merge(feature_df, df, how='left',  left_on="date", right_on="date_short")
        feature_df = feature_df.drop(['date_short'], axis=1)
        #scale the feature between -1 and 1
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        newer_column_name = new_column_name+"_scaled"
        newer_column_name_1 = new_column_name_1+"_scaled"
        feature_df[[newer_column_name, newer_column_name_1]] = min_max_scaler.fit_transform(feature_df[[new_column_name, new_column_name_1]])


    else:
        df = twitter_df.groupby('date_short', as_index=False)[sentiment_str].mean()
        df.date_short = pd.to_datetime(df.date_short)
        feature_df = pd.merge(feature_df, df, how='left', left_on='date', right_on='date_short')
        feature_df= feature_df.drop(['date_short'], axis=1)
        new_column_name = set_description+"_"+sentiment_str
        feature_df = feature_df.rename(columns={sentiment_str: new_column_name})
        #scale the feature between -1 and 1
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        newer_column_name = new_column_name+"_scaled"
        feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def average_sentiment(feature_df, set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    sent_str_11 = set_description+"_"+sent_str_1+"_scaled"
    sent_str_22 = set_description+"_"+sent_str_2+"_scaled"
    sent_str_33 = set_description+"_"+sent_str_3+"_scaled"
    df = feature_df.loc[:, ["date",sent_str_11, sent_str_22, sent_str_33]]
    df['average_sentiment_normalized'] = df.iloc[:, 1:4].mean(axis=1)
    feature_df = pd.merge(feature_df, df[['date','average_sentiment_normalized']], how='left', on='date')

    return feature_df
#----------------------------
def sentiment_volatility(twitter_df, feature_df,set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    twitter_df = get_expectation_value_column(twitter_df,sent_str_3, set_description)
    sent_str_33 = sent_str_3+"_expectation_value"
    array = get_sentiment_array(sent_str_1, sent_str_2, sent_str_33, 0,0)
    unique_dates = twitter_df['date_short'].unique()
    df = pd.DataFrame({'date_short': unique_dates})
    for s in array:
        new_column_name = set_description+"_"+s+"_sentiment_volatility"
        for index, row in df.iterrows():
            date = row['date_short']
            std = twitter_df.loc[twitter_df['date_short']== date,s].std()
            volatility = std**.5
            df.at[index, new_column_name] =  volatility
        feature_df = pd.merge(feature_df, df, how='left',  left_on="date", right_on="date_short")
        df = df.drop([new_column_name], axis=1)
        feature_df = feature_df.drop(['date_short'], axis=1)
        #scale the feature between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
        newer_column_name = new_column_name+"_scaled"
        feature_df[[newer_column_name]] = min_max_scaler.fit_transform(feature_df[[new_column_name]])

    return feature_df
#----------------------------
def sentiment_ROC(twitter_df, feature_df, n,set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), n is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    array = get_sentiment_array(sent_str_1, sent_str_2, sent_str_3, set_description,'scaled')
    roc_df = feature_df['date']
    roc_df = pd.Series(feature_df['date'], name = "date")
    for s in array:
            df = feature_df[s]
            Diff = df.diff(n - 1)
            Shift = df.shift(n - 1)
            ROC = pd.Series(((Diff / Shift) * 100), name = "ROC_"+str(n)+"_"+s)
            roc_df=pd.concat([roc_df,ROC],axis=1)

    feature_df = pd.merge(feature_df, roc_df, how='left',  on="date")

    return feature_df
#----------------------------
def sentiment_MOM(twitter_df, feature_df, n,set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    array = get_sentiment_array(sent_str_1, sent_str_2, sent_str_3, set_description,'scaled')
    mom_df = feature_df['date']
    mom_df = pd.Series(feature_df['date'], name = "date")
    for s in array:
            df = feature_df[s]
            MOM = pd.Series(df.diff(n), name='Momentum_'+str(n)+"_"+s)
            mom_df=pd.concat([mom_df,MOM],axis=1)

    feature_df = pd.merge(feature_df, mom_df, how='left',  on="date")

    return feature_df
#----------------------------
def sentiment_RSI(twitter_df, feature_df, n,set_description):
    """
    - Parameters: twitter_df & feature_df (Both df), d is integer which determines the "look.back-period"
    - Returns: df_final, same shape as df but with the inputed features
    """
    array = get_sentiment_array(sent_str_1, sent_str_2, sent_str_3, set_description,'scaled')
    rsi_df = feature_df['date']
    rsi_df = pd.Series(feature_df['date'], name = "date")
    for s in array:
        series = feature_df[s]
        #calculation of relative strength index
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[n-1]] = np.mean( u[:n] ) #first value is sum of avg gains
        u = u.drop(u.index[:(n-1)])
        d[d.index[n-1]] = np.mean( d[:n] ) #first value is sum of avg losses
        d = d.drop(d.index[:(n-1)])
        #rs = pd.stats.moments.ewma(u, com=n-1, adjust=False)
        rs=pd.DataFrame.ewm(u,com=n-1,adjust=False).mean()
        pd.DataFrame.ewm(d,com=n-1,adjust=False).mean()
        #pd.stats.moments.ewma(d, com=n-1, adjust=False)
        RSI = pd.Series((100 - 100 / (1 + (rs))), name='RSI_'+str(n)+"_"+s)
        rsi_df=pd.concat([rsi_df,RSI],axis=1)

    feature_df = pd.merge(feature_df, rsi_df, how='left',  on="date")

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
# Helper Functions
#---------------------
def get_sentiment_array(sent_str_1, sent_str_2, sent_str_3, set_description,scale_description):
    if scale_description == 0:
        if set_description == 0:
            sent_array = [sent_str_1, sent_str_2, sent_str_3]
        else:
            sent_str_1 = set_description+"_"+sent_str_1
            sent_str_2 = set_description+"_"+sent_str_2
            sent_str_3 = set_description+"_"+sent_str_3
            sent_array = [sent_str_1, sent_str_2, sent_str_3]
    else:
        if set_description == 0:
            sent_str_1 = sent_str_1+"_scaled"
            sent_str_2 = sent_str_2+"_scaled"
            sent_str_3 = sent_str_3+"_scaled"
            sent_array = [sent_str_1, sent_str_2, sent_str_3]
        else:
            sent_str_1 = set_description+"_"+sent_str_1+"_scaled"
            sent_str_2 = set_description+"_"+sent_str_2+"_scaled"
            sent_str_3 = set_description+"_"+sent_str_3+"_scaled"
            sent_array = [sent_str_1, sent_str_2, sent_str_3]
    return sent_array
#---------------------
def get_expectation_value_column(twitter_df, sentiment_str, set_description):
    column_name = sentiment_str+"_prob"
    number_column_name = sentiment_str+"_number"
    twitter_df[number_column_name] = twitter_df[sentiment_str]
    twitter_df[number_column_name].replace({"NEG": -1, "NEU": 0, "POS": 1}, inplace=True)
    column_name_expectation = sentiment_str+"_expectation_value"
    twitter_df[column_name_expectation] = twitter_df[number_column_name]*twitter_df[column_name]

    return twitter_df
