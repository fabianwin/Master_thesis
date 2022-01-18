import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import time
import re
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from pytrends.request import TrendReq
import urllib3
import random
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#############Global Parameters###################
#set global parameters for twitterSearchscraper
maxTweets = 10000000000000
restrictions='min_faves:5 exclude:retweets lang:"en"' #min. 5 likes ,no retweets, in english
dates = ['2021', '2020', '2019', '2018', '2017']
ticker_col =["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count"]
product_col =["tweet_id","date_short","username","content","likes","retweets","followers Count","keyword"]
ticker_tweets_df = pd.DataFrame(columns=ticker_col)
product_tweets_df = pd.DataFrame(columns=product_col)

#############Functions###################
def scrape_tweets(keyword,date, year,twitter_df,ticker_col):
    """
    - Parameters: doc (a Stanza Document object)
    - Returns: a mean sentiment score
    """
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
            if i == maxTweets :
                break
            tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=ticker_col)
            if isinstance(tmp[5], str) == True:
                twitter_df = twitter_df.append( tmp, ignore_index=True)
            print(tweet.date)

    twitter_df.dropna(axis=0, how="any")
    #print("Unique Users: ",twitter_df['username'].nunique(),"/", twitter_df.shape[0])

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/ticker_sets')
    my_file = 'twitter_set_'+keyword+"_"+year+".csv"
    twitter_df.to_csv(os.path.join(my_path, my_file))

    return twitter_df

#---------------------
def get_ticker_tweets(keyword):
    pdList = []
    for year in dates:
        start = 'since:'+year+'-01-01'
        end = ' until:'+year+'-12-31'
        datum = start+end
        df = scrape_tweets(keyword, datum, year, ticker_tweets_df, ticker_col)
        pdList.append(df)

    entire_twitter_df = pd.concat(pdList)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'ticker_set_'+keyword+".csv"
    entire_twitter_df.to_csv(os.path.join(my_path, my_file))

#---------------------
def date_to_epoch_intervall(date):
    date_time_1 = str(date)+' 00:00:00-GMT'
    date_time_2 = str(date)+' 23:59:59-GMT'
    pattern = '%Y-%m-%d %H:%M:%S-%Z'
    epoch_1 = time.mktime(time.strptime(date_time_1, pattern))
    epoch_2 = time.mktime(time.strptime(date_time_2, pattern))
    epoch_intervall = 'since_time:'+str(epoch_1)[0:10]+' until_time:'+str(epoch_2)[0:10]

    return epoch_intervall

#---------------------
def scrape_google_trendwords(keyword):
    for year in dates:
        sdate = date(int(year),1,1)
        edate = date(int(year),12,31)
        full_date_list = pd.date_range(sdate,edate-timedelta(days=1),freq='d')

        #build the trendreq payload
        pytrend = TrendReq(retries=2, backoff_factor=0.1, requests_args={'verify':False})
        #provide your search terms
        kw_list=[keyword]
        #for every day we have a tweet in the ticker_set we look up the corresponding related queries on that day
        complete = pd.DataFrame()
        #build complete (set with all the related search words from google)
        for i in full_date_list:
            i = str(i.date())
            date_local = i+" "+i
            pytrend.build_payload(kw_list,timeframe=date_local, cat=0, geo='', gprop='')
            #get related queries
            related_queries = pytrend.related_queries()
            #construct df with trend words
            rising = list(related_queries.values())[0]['rising']
            if rising is not None:
                #print(rising)
                rising['date'] = i
                complete = complete.append(rising, ignore_index= True)
                print(rising)
            time.sleep(7+random.random())

        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets')
        my_file = 'keyword_set_'+keyword+"_"+year+".csv"
        complete.to_csv(os.path.join(my_path, my_file))

        print("Google related queries were scraped and we know all the keywords for scraping the",keyword,"Product set in the year", year)
        print("there are", complete.shape[0]," unique combinations of keywords and dates")

#---------------------
def filter_google_queries(keyword,df_tweets):
    len_list = list()
    #load datasets
    semantic_search_words = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets/keyword_filter.csv', usecols=[0], encoding='utf-8')
    neg_exp = list(semantic_search_words['neg_words'].values.flatten())
    #add special character strings
    special_char = ['kaç tl','dólar','fiyatı','árfolyam', 'kaç','cotação','giá', 'fiyat grafi_i','yatƒ±rma','soru≈üturma',' –±–∏—Ä–∂–∞','ka√ß','fiyatƒ±','–∫—É—Ä—Å','davasƒ±']
    neg_exp = neg_exp+special_char
    # make a string with separator | from the list
    neg_exp_string = "|".join(neg_exp)

    # replace german word with english translation, choose Stock since this is the only relevant keyword which needs translation
    df_tweets['query'] = df_tweets['query'].replace("aktie","stock",regex=True)
    # drop duplicates
    df_tweets.drop_duplicates(subset=['query','date'], keep='first', inplace=True)
    df_tweets = df_tweets[~df_tweets['query'].str.contains(neg_exp_string, case=False)]
    print('Shape after negative word filter: ' + str(len(df_tweets)))
    len_list.append(len(df_tweets))

    return df_tweets

#---------------------
def scrape_product_tweets(keyword):
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'keyword_set_'+keyword+".csv"
    keyword_df = pd.read_csv(os.path.join(my_path, my_file))


    print(keyword_df.shape)
    keyword_df = keyword_df.dropna(axis=0, how="any")
    print(keyword_df.shape)
    keyword_df = filter_google_queries(keyword, keyword_df)
    print(keyword_df.shape)


    #iterate over every row in the complete-set
    df = product_tweets_df
    for n,row in keyword_df.iterrows():
        date_of_keyword = date_to_epoch_intervall(row['date'])
        related_keyword = row['query']
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(related_keyword+' '+date_of_keyword+' '+restrictions).get_items()):
            if i==maxTweets:
                break
            tmp = pd.Series([tweet.id, tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount, related_keyword], index=product_tweets_df.columns)
            print(tweet.date)
            df = df.append( tmp, ignore_index=True)

    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'product_set_'+keyword+".csv"
    df.to_csv(os.path.join(my_path, my_file))

    print("Scraped all tweets based on the received trendwords for", keyword)
#---------------------
