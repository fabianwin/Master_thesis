#Construct the Ticker Set for TSLA
import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import datetime

col =["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count"]
twitter_df = pd.DataFrame(columns=col)


maxTweets = 10000000
keyword = "Bitcoin"
date = 'since:2021-12-30 until:2021-12-31'
restrictions='min_faves:1000 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english

for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
        if i == maxTweets :
            break
        tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=col)
        if isinstance(tmp[5], str) == True:
            twitter_df = twitter_df.append( tmp, ignore_index=True)
        print(tweet.date)

twitter_df.dropna(axis=0, how="any")
print(twitter_df.shape)

twitter_df.to_csv(r'Data/test.csv')
