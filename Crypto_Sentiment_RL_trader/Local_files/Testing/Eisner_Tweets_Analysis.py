import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,f1_score, confusion_matrix
from functions import preprocess, perform_sentiment_analysis, normalize, round

"""
#step 1: preprocess
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_original.csv')
eisner_tweets = preprocess(eisner_tweets)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_preprocessed.csv', index = False)


#step 2: calculate Sentiment scores
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_preprocessed.csv')
print(eisner_tweets.dtypes)
eisner_tweets = perform_sentiment_analysis(eisner_tweets)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_sentiment_scores.csv', index = False)
"""

#step 3: adjust scores and calculate scores
#load dataframe
eisner_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_sentiment_scores.csv')
#Angleichung Stanford_sentiment
eisner_tweets = round(eisner_tweets, 'Stanford_sentiment')
eisner_tweets["Stanford_sentiment rounded"].replace({0: -1, 1: 0, 2:1}, inplace=True)
#Angleichung TextBlob_sentiment
eisner_tweets = round(eisner_tweets, 'TextBlob_sentiment')
#Angleichung Flair_sentiment
eisner_tweets = normalize(eisner_tweets, 'Flair_sentiment')
eisner_tweets = round(eisner_tweets, 'Flair_sentiment normalized')
#Angleichung finiteautomata_sentiment
eisner_tweets["finiteautomata_sentiment"].replace({"NEG": -1, "NEU": 0, "POS":1}, inplace=True)
#Angleichung cardiffnlp_sentiment
eisner_tweets["cardiffnlp_sentiment"].replace({"negative": -1, "neutral": 0, "positive":1}, inplace=True)
eisner_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/Tests/eisner_tweets_sentiment_scores_adjusted.csv', index = False)


y_true = eisner_tweets['label']
y_stanford = eisner_tweets['Stanford_sentiment rounded']
y_textblob = eisner_tweets['TextBlob_sentiment rounded']
y_flair = eisner_tweets['Flair_sentiment normalized rounded']
y_finiteautomata = eisner_tweets['finiteautomata_sentiment']
y_cardiffnlp = eisner_tweets['cardiffnlp_sentiment']


print(f1_score(y_true, y_stanford, average="macro"))
print(f1_score(y_true, y_textblob, average="macro"))
print(f1_score(y_true, y_flair, average="macro"))
print("y_finiteautomata",f1_score(y_true, y_finiteautomata, average="macro"))
print("cardiffnlp", f1_score(y_true, y_cardiffnlp, average="macro"))
print("  ")

print(accuracy_score(y_true, y_stanford))
print(accuracy_score(y_true, y_textblob))
print(accuracy_score(y_true, y_flair))
print(accuracy_score(y_true, y_finiteautomata))
print(accuracy_score(y_true, y_cardiffnlp))
print("  ")

print(confusion_matrix(y_true, y_stanford))
print("  ")

print(confusion_matrix(y_true, y_finiteautomata))
print("  ")

print(confusion_matrix(y_true, y_cardiffnlp))

#print(eisner_tweets.groupby("Stanford_sentiment rounded").count())
