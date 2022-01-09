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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#############Global Parameters###################
#set global parameters for twitterSearchscraper
maxTweets = 10000000000000
restrictions='min_faves:50000 exclude:retweets lang:"en"' #min. 10 likes ,no retweets, in english
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

    my_path = os.path.normpath(r'Crypto_Sentiment_RL_trader/EULER_files/2.Data_collection/1.Twitter_Scrapping/Data/ticker_sets')
    #r'Master_thesis/Crypto_Sentiment_RL_trader/EULER_files/2.Data_collection/1.Twitter_Scrapping/Data/ticker_sets')
    my_file = 'twitter_set_'+keyword+"_"+year+".csv"

    print(os.path.join(my_path, my_file))
    twitter_df.to_csv((r'Crypto_Sentiment_RL_trader/EULER_files/2.Data_collection/1.Twitter_Scrapping/Data/ticker_sets/test.csv'))
    
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
def scrape_product_tweets(keyword):
    pdList = []
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

        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets')
        my_file = 'keyword_set_'+keyword+"_"+year+".csv"
        complete.to_csv(os.path.join(my_path, my_file))


        print("Google related queries were scraped and we know all the keywords for scraping the",keyword,"Product set in the year", year)
        print("there are", complete.shape[0]," unique combinations of keywords and dates")

        #iterate over every row in the complete-set
        df = product_tweets_df
        for n,row in complete.iterrows():
            date_in_complete = date_to_epoch_intervall(row['date'])
            related_keyword = row['query']
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(related_keyword+' '+ date_in_complete+' '+restrictions).get_items()):
                if i==maxTweets:
                    break
                tmp = pd.Series([tweet.id, tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount, related_keyword], index=product_tweets_df.columns)
                #tmp.date = str(tmp.date)[0:10]
                df = df.append( tmp, ignore_index=True)

        print("Scraped all tweets based on the received trendwords from year", year)
        print(" ")

        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/product_sets')
        my_file = 'product_set_'+keyword+"_"+year+".csv"
        df.to_csv(os.path.join(my_path, my_file))
        pdList.append(df)

    entire_twitter_df = pd.concat(pdList)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'product_set_'+keyword+".csv"
    entire_twitter_df.to_csv(os.path.join(my_path, my_file))

#---------------------
