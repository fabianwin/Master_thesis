from pytrends.request import TrendReq
import numpy as np

#create the dataframe for the sub set, columns are the 2 changing variables for twitter scrapers
col =["keyword","date"]
sub_set = pd.DataFrame(columns=col)

#load the ticker tweets
ticker_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')

#get the unqiue dates, for each date we get related queries to Stock ticker
unique_dates = ticker_tweets['date_short'].unique()

#build the trendreq payload
pytrends = TrendReq()
trend_word = ["GM"]

#for every day we have a tweet in the ticker_set we look up the corresponding related queries on that day
for i in unique_dates:
    i = str(i)
    date = i+" "+i
    pytrends.build_payload(trend_word, cat=0, timeframe=date, geo='', gprop='')
    data = pytrends.related_queries()

    if data['GM']['top'] is None:
        print("dict is empty")
    else:
        data =  pd.DataFrame.from_dict({(i): data['GM']['top'][i]
                        for i in data['GM']['top'].keys()},
                        orient ='columns')

        for index, row in data.iterrows():
            keyword = row['query']
            datum = i
            tmp = pd.Series([keyword, datum], index=['keyword', 'date'])
            sub_set = sub_set.append( tmp, ignore_index=True )

#----------------------------
print("Google related queries are scraped and we know all the keywords for scraping the Product set")
print("there are", len(unique_dates), " unique dates and ", sub_set.shape[0]," unique combinations of keywords and dates")
print("now find the tweets on the given day with the related query")

#----------------------------
from datetime import datetime
import time

def date_to_epoch_intervall(date):
    """
    - Parameters: date in the format yyyy-mm-dd
    - Returns: Epoch intervall for the entire input day
    """
    date_time_1 = str(date)+' 00:00:00-GMT'
    date_time_2 = str(date)+' 23:59:59-GMT'
    pattern = '%Y-%m-%d %H:%M:%S-%Z'
    epoch_1 = time.mktime(time.strptime(date_time_1, pattern))
    epoch_2 = time.mktime(time.strptime(date_time_2, pattern))
    epoch_intervall = 'since_time:'+str(epoch_1)[0:10]+' until_time:'+str(epoch_2)[0:10]

    return epoch_intervall

#create the dataframe for the product set, same columns like ticker set
col =["tweet_id","date","username","content","likes","retweets","followers Count","keyword"]
product_tweets = pd.DataFrame(columns=col)
keywords = sub_set['keyword'].tolist()
dates = sub_set['date'].tolist()

#set global parameters for twitterSearchscraper
maxTweets = 100
restrictions='min_faves:100 exclude:retweets lang:"en"' #min. 100 likes ,no retweets, in english

#iterate over every row in the sub_set
for n,keyword in enumerate(keywords):
    date = date_to_epoch_intervall(dates[n])
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(keyword+' '+ date +' '+restrictions).get_items()):
        if i==maxTweets:
            break
        tmp = pd.Series([tweet.id, tweet.date, tweet.user.username, tweet.content, tweet.likeCount,tweet.retweetCount,tweet.user.followersCount, keyword], index=product_tweets.columns)
        tmp.date = str(tmp.date)[0:10]
        product_tweets = product_tweets.append( tmp, ignore_index=True)


#---------------------------
print("Product Set completely scraped")
product_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_tweets_GM.csv', index = False)
