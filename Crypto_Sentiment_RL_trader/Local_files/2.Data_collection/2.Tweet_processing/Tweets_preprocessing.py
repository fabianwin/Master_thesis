import pandas as pd
import numpy as np
import os
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_tweets(df):
    for n,row in df.iterrows():
        tweet = row['content']
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#'
        tweet = re.sub(r'\@\w+|\#','', tweet)
        # Remove consequtive question marks
        tweet = re.sub('[?]+[?]', ' ', tweet)
        # Remove &amp - is HTML code for hyperlink
        tweet = re.sub(r'\&amp;','&', tweet)
        #Replace emoji with text
        tweet = emoji.demojize(tweet, language="en", delimiters=(" ", " "))
        df.at[n,'content'] =  tweet

    return df

#################Ticker Sets##################
coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list_ticker:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'ticker_set_'+coin+".csv"

    df = pd.read_csv(os.path.join(my_path, my_file))
    df = df.dropna(axis=0, how='any', inplace=False)
    df = preprocess(df)
    df = preprocess_tweets(df)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'ticker_set_preprocessed_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))

print("All Ticker sets are preprocessed")

#################news Sets##################
coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list_news:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'news_set_'+coin+".csv"

    df = pd.read_csv(os.path.join(my_path, my_file))
    df = df.dropna(axis=0, how='any', inplace=False)
    df = preprocess(df)
    df = preprocess_tweets(df)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'news_set_preprocessed_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))
