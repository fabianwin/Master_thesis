import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from pysentimiento.preprocessing import preprocess_tweet
import os

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
        tweet = emoji.demojize(text, language="en", delimiters=(" ", " "))
        df.at[n,'content'] =  tweet

    return df
#----------------------------------------------
def preprocess(df):
    for n,row in df.iterrows():
        tweet = row['content']
        tweet = preprocess_tweet(tweet,lang="en")
        df.at[n,'content'] =  tweet
        print(n)

    return df
#----------------------------
df =pd.read_csv(r'/Users/fabian/Downloads/texts.csv')
print(df)
