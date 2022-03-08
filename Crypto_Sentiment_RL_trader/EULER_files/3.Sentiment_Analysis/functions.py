from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import stanza
import re
from scipy.special import softmax
import csv
import urllib.request
import flair
from flair.data import Sentence
from textblob import TextBlob


def get_stanford_sentiment(df):
    """
    - Parameters: df, df has all the tweets info stored with preprocessed tweets
    - Returns: df_final, same shape as df but with the calculated scores
    """
    # Download an English model into the default directory
    print("Downloading English model...")
    stanza.download('en')
    # Build an English pipeline, with all processors by default
    print("Building an English pipeline...")
    en_nlp = stanza.Pipeline('en')

    df['Stanford_sentiment'] = np.nan
    for index, row in df.iterrows():
        text = row['content']
        text = en_nlp(text)
        score = sentence_sentiment_df(text)
        df.at[index,'Stanford_sentiment'] =  score

    return df

#----------------------------------------------
def get_textblob_sentiment(df):
    """
    - Parameters: df, df has all the tweets info stored with preprocessed tweets
    - Returns: df_final, same shape as df but with the calculated scores
    """
    df['TextBlob_sentiment'] = np.nan
    for index, row in df.iterrows():
        text = row['content']
        score = TextBlob(text).sentiment.polarity
        df.at[index,'TextBlob_sentiment'] =  score

    return df

#----------------------------------------------
def get_flair_sentiment(df):
    """
    - Parameters: df, df has all the tweets info stored with preprocessed tweets
    - Returns: df_final, same shape as df but with the calculated scores
    """

    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    df['Flair_sentiment'] = np.nan
    for index, row in df.iterrows():
        sent=0
        sentence = flair.data.Sentence(row['content'])
        sentiment_model.predict(sentence)
        if sentence.labels[0].value=="POSITIVE":
            sent = sentence.labels[0].score
        else :
            sent =1-sentence.labels[0].score
        df.at[index,'Flair_sentiment'] =  sent

    return df

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
def get_finiteautomata_sentiment(df):
    analyzer = create_analyzer(task="sentiment", lang="en")
    df['finiteautomata_sentiment'] = "NaN"
    df['finiteautomata_sentiment_prob'] = "NaN"
    for index, row in df.iterrows():
        text = row['content']
        result = analyzer.predict(text)
        prob = max(list(result.probas.values()))
        out = result.output
        df.at[index,'finiteautomata_sentiment'] =  out
        df.at[index,'finiteautomata_sentiment_prob'] =  prob

    return df

#----------------------------------------------
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

#----------------------------------------------
def get_cardiffnlp_sentiment(df):
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    df['cardiffnlp_sentiment'] = "NaN"
    for index, row in df.iterrows():
        text = row['content']
        print(type(text))
        text = preprocess(text)
        print(type(text))

        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        best = np.argmax(scores)
        label = labels[best]
        score = scores[best] #probability that label is correct

        df.at[index,'cardiffnlp_sentiment'] =  label

    return df

#----------------------------------------------
def normalize(df, col_name_str):
    data = df[col_name_str]
    data = (2*(data - np.min(data)) / (np.max(data) - np.min(data)))-1 #normalize between [-1,1]
    new_col_name = col_name_str+" "+"normalized"
    df[new_col_name]=data
    return df

#----------------------------------------------
def round(df, col_name_str):
    data = df[col_name_str]
    data =  data.round()
    new_col_name = col_name_str+" "+"rounded"
    df[new_col_name]=data
    return df
