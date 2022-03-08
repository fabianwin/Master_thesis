import pandas as pd
import numpy as np
import re
import emoji
from sklearn.metrics import accuracy_score,precision_score,f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from functions import normalize, round
from functions import get_stanford_sentiment, get_textblob_sentiment, get_flair_sentiment, get_finiteautomata_sentiment, get_cardiffnlp_sentiment

#----------------------------------------------
def perform_sentiment_analysis(df):
    df = get_textblob_sentiment(df)
    df = get_flair_sentiment(df)
    df = get_cardiffnlp_sentiment(df)
    df = get_finiteautomata_sentiment(df)

    return df

#----------------------------------------------
def preprocess(df):
    for n,row in df.iterrows():
        tweet = row['content']
        tweet = preprocess_tweet(tweet,lang="en")
        df.at[n,'content'] =  tweet
        print(n)

    return df

#----------------------------------------------
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

#step 1: preprocess
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_original.csv')
eisner_tweets = preprocess_tweets(eisner_tweets)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_preprocessed.csv', index = False)

#step 2: calculate Sentiment scores
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_preprocessed.csv')
eisner_tweets = perform_sentiment_analysis(eisner_tweets)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_sentiment_scores.csv', index = False)

#step 3: adjust and calculate scores
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_sentiment_scores.csv')
#uniform handling of different NLP models for comparison
eisner_tweets = round(eisner_tweets, 'TextBlob_sentiment')
eisner_tweets = normalize(eisner_tweets, 'Flair_sentiment')
eisner_tweets = round(eisner_tweets, 'Flair_sentiment normalized')
eisner_tweets["finiteautomata_sentiment"].replace({"NEG": -1, "NEU": 0, "POS":1}, inplace=True)
eisner_tweets["cardiffnlp_sentiment"].replace({"negative": -1, "neutral": 0, "positive":1}, inplace=True)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_sentiment_scores_adjusted.csv', index = False)

#step 4: Model comparison
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/eisner_tweets_sentiment_scores_adjusted.csv')
y_true = eisner_tweets['label']
y_textblob = eisner_tweets['TextBlob_sentiment rounded']
y_flair = eisner_tweets['Flair_sentiment normalized rounded']
y_finiteautomata = eisner_tweets['finiteautomata_sentiment']

print("f1 scores")
print("y_textblob",f1_score(y_true, y_textblob, average="macro"))
print("y_flair",f1_score(y_true, y_flair, average="macro"))
print("y_finiteautomata",f1_score(y_true, y_finiteautomata, average="macro"))
print("  ")
print("accuracy")
print("y_textblob",accuracy_score(y_true, y_textblob))
print("y_flair",accuracy_score(y_true, y_flair))
print("y_finiteautomata",accuracy_score(y_true, y_finiteautomata))
print("  ")
print("y_textblob")
print(confusion_matrix(y_true, y_textblob))
print("y_flair")
print(confusion_matrix(y_true, y_flair))
print("y_finiteautomata")
print(confusion_matrix(y_true, y_finiteautomata))
print("  ")

"""
cm = confusion_matrix(y_true, y_finiteautomata)
f = sns.heatmap(cm, annot=True, )
plt.savefig(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/fig.png')
"""
