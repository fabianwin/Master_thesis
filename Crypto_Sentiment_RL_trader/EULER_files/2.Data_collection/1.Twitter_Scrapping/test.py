import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import datetime


maxTweets = 100
keyword = "BTC"
date = 'since:2021-04-06 until:2021-04-08'
restrictions='min_faves:1 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english
col =["tweet_id","date_short","date_medium","date_long","username","content","likes","retweets","followers Count"]
twitter_df = pd.DataFrame(columns=col)


for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword +' '+ date+' '+restrictions).get_items()) :
        if i == maxTweets :
            break
        tmp = pd.Series([tweet.id,tweet.date.date(),tweet.date.replace(minute=0, second=0),tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount], index=col)
        print(tmp.content)
        if isinstance(tmp[5], str) == True:
            twitter_df = twitter_df.append( tmp, ignore_index=True)
