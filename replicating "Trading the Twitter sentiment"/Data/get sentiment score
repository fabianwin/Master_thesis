#preprocess the maxTweets
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

#---------------------
import numpy as np
import flair
from flair.data import Sentence

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

# Calculate TSLA tweets
ticker_tweets_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv')
product_tweets_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_tweets_TSLA.csv.csv')

Stanford_sentiment = []
for index, row in ticker_tweets_TSLA.iterrows():
    text = Sentence(preprocess_tweet(row['content']))
    text = str(text)
    text = en_nlp(text)
    score = sentence_sentiment_df(text)
    Stanford_sentiment.append(score)

ticker_tweets_TSLA['Stanford_sentiment'] = Stanford_sentiment
ticker_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv', index = False)

Stanford_sentiment = []
for index, row in product_tweets_TSLA.iterrows():
    text = Sentence(preprocess_tweet(row['content']))
    text = str(text)
    text = en_nlp(text)
    score = sentence_sentiment_df(text)
    Stanford_sentiment.append(score)

product_tweets_TSLA['Stanford_sentiment'] = Stanford_sentiment
product_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_tweets_TSLA.csv', index = False)
print("TESLA sentiments score are calculated")

# Calculate GM tweets
ticker_tweets_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
product_tweets_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_tweets_GM.csv.csv')

Stanford_sentiment = []
for index, row in ticker_tweets_GM.iterrows():
    text = Sentence(preprocess_tweet(row['content']))
    text = str(text)
    text = en_nlp(text)
    score = sentence_sentiment_df(text)
    Stanford_sentiment.append(score)

ticker_tweets_GM['Stanford_sentiment'] = Stanford_sentiment
ticker_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv', index = False)

Stanford_sentiment = []
for index, row in product_tweets_GM.iterrows():
    text = Sentence(preprocess_tweet(row['content']))
    text = str(text)
    text = en_nlp(text)
    score = sentence_sentiment_df(text)
    Stanford_sentiment.append(score)

product_tweets_GM['Stanford_sentiment'] = Stanford_sentiment
product_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_tweets_GM.csv', index = False)
print("GM sentiments score are calculated")
