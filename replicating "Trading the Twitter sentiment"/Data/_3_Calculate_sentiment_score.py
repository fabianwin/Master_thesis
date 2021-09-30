#preprocess the maxTweets
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import flair
from flair.data import Sentence
from textblob import TextBlob


def preprocess_tweet(tweet):
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#'
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove consequtive question marks
    tweet = re.sub('[?]+[?]', ' ', tweet)

    return tweet

#---------------------
#Run this Cell to download the appropriate model and start it
import stanza
# Download an English model into the default directory
print("Downloading English model...")
stanza.download('en')

# Build an English pipeline, with all processors by default
print("Building an English pipeline...")
en_nlp = stanza.Pipeline('en')


#----------------------------------------------
def get_stanford_sentiment(twitter_df):
    """
    - Parameters: twitter_df, twitter_df has all the tweets info stored
    - Returns: df_final, same shape as twitter_df but with the calculated scores
    """

    Stanford_sentiment = []
    for index, row in twitter_df.iterrows():
        text = Sentence(preprocess_tweet(row['content']))
        text = str(text)
        text = en_nlp(text)
        score = sentence_sentiment_df(text)
        Stanford_sentiment.append(score)
    twitter_df['Stanford_sentiment'] = Stanford_sentiment

    return twitter_df

#----------------------------------------------
def get_textblob_sentiment(twitter_df):
    """
    - Parameters: twitter_df, twitter_df has all the tweets info stored
    - Returns: df_final, same shape as twitter_df but with the calculated scores
    """

    Textblob_sentiment = []
    for index, row in twitter_df.iterrows():
        text = Sentence(preprocess_tweet(row['content']))
        text = str(text)
        score = TextBlob(text).sentiment.polarity
        Textblob_sentiment.append(score)
    twitter_df['TextBlob_sentiment'] = Textblob_sentiment

    return twitter_df

#----------------------------------------------
def get_flair_sentiment(twitter_df):
    """
    - Parameters: twitter_df, twitter_df has all the tweets info stored
    - Returns: df_final, same shape as twitter_df but with the calculated scores
    """

    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    probs = []
    sentiments = []
    flair_sentiment=[]
    for index, row in twitter_df.iterrows():
        sent=0
        sentence = flair.data.Sentence(row['content'])
        sentiment_model.predict(sentence)
        # extract sentiment prediction
        probs.append(sentence.labels[0].score)  # numerical score 0-1
        sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'
        if sentence.labels[0].value=="POSITIVE":
            sent = sentence.labels[0].score
        else :
            sent =1-sentence.labels[0].score
        flair_sentiment.append(sent)
    twitter_df['Flair_sentiment'] = flair_sentiment

    return twitter_df
#----------------------------------------------
def sentence_sentiment_df(doc):
    """
    - Parameters: doc (a Stanza Document object)
    - Returns: a mean sentiment score
    """
    sentiment_values = []
    for sentence in doc.sentences:
        sentiment_score = sentence.sentiment
        sentiment_values.append(sentiment_score)

    mean_sentiment = np.mean(sentiment_values)
    return mean_sentiment
#----------------------------------------------



# Calculate TSLA tweets
ticker_tweets_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv')
product_tweets_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_TSLA.csv')

get_stanford_sentiment(ticker_tweets_TSLA)
get_textblob_sentiment(ticker_tweets_TSLA)
get_flair_sentiment(ticker_tweets_TSLA)

get_stanford_sentiment(product_tweets_TSLA)
get_textblob_sentiment(product_tweets_TSLA)
get_flair_sentiment(product_tweets_TSLA)

ticker_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv', index = False)
product_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_TSLA.csv', index = False)

print("TESLA sentiments score are calculated")


# Calculate GM tweets
ticker_tweets_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
product_tweets_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_GM.csv')

get_stanford_sentiment(ticker_tweets_GM)
get_textblob_sentiment(ticker_tweets_GM)
get_flair_sentiment(ticker_tweets_GM)

get_stanford_sentiment(product_tweets_GM)
get_textblob_sentiment(product_tweets_GM)
get_flair_sentiment(product_tweets_GM)

ticker_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv', index = False)
product_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_GM.csv', index = False)

print("GM sentiments score are calculated")
