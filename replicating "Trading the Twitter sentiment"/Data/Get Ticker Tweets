#Construct the Ticker Set for TSLA
import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import datetime


col =["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count"]
ticker_tweets = pd.DataFrame(columns=col)
ticker_tweets_TSLA = ticker_tweets
ticker_tweets_GM = ticker_tweets


#Get Tesla tweets
maxTweets = 100000  # the number of tweets you require
keyword = "TSLA"
date = 'since:2020-08-01 until:2021-08-30'
restrictions='min_faves:100 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english

for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
        if i == maxTweets :
            break
        tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=ticker_tweets.columns)
        ticker_tweets_TSLA = ticker_tweets_TSLA.append( tmp, ignore_index=True )

ticker_tweets_TSLA.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv', index = False)

#----------------------
#Get Ford tweets
maxTweets = 100000  # the number of tweets you require
keyword = "GM"
date = 'since:2020-08-01 until:2021-08-30'
restrictions='min_faves:100 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english

for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
        if i == maxTweets :
            break
        tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=ticker_tweets.columns)
        ticker_tweets_GM = ticker_tweets_GM.append( tmp, ignore_index=True )

ticker_tweets_GM.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv', index = False)
#---------------------

ticker_tweets_GM.head(5)
print("Ticker Set completely scraped")
