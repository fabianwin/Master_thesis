from functions import scrape_product_tweets
import pandas as pd

#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
#for product sets search for written out terms, than iterate through query findings
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA INU']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins


#scrape_product_tweets('DOGECOIN')


#1. Filter keyword set for unnecessary word
# Load the words to be filtered for and make a list from it
len_list = list()
semantic_search_words = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets/keyword_filter.csv', usecols=[0])
neg_exp = list(semantic_search_words['neg_words'].values.flatten())
special_char = ['kaç tl','dólar','fiyatı','árfolyam', 'kaç','cotação','giá', 'fiyat grafi_i','yatƒ±rma','soru≈üturma',' –±–∏—Ä–∂–∞','ka√ß','fiyatƒ±','–∫—É—Ä—Å','davasƒ±']
neg_exp = neg_exp+special_char
neg_exp = [w.replace('aktie', 'stock') for w in neg_exp]
# make a string with separator | from the list
neg_exp_string = "|".join(neg_exp)

# make new column for data frame which contains boolean if the row is relevant
df_tweets = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/keyword_set_DOGECOIN.csv', encoding='utf-8')
# replace german words
df_tweets['query'] = df_tweets['query'].replace("aktie","stock",regex=True)
# drop duplicates
df_tweets.drop_duplicates(subset=['query','date'], keep='first', inplace=True)
df_tweets = df_tweets[~df_tweets['query'].str.contains(neg_exp_string, case=False)]
df_tweets.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets/filter1.csv')
print('Shape after negative word filter: ' + str(len(df_tweets)))
len_list.append(len(df_tweets))
