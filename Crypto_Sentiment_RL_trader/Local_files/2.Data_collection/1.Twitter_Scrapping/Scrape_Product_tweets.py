from functions import scrape_product_tweets

#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
#for product sets search for written out terms, than iterate through query findings
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA_INU']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

#coins with date ordered in the strucutre: %Y-%m-%d
ymd_coins = ['BITCOIN', 'CARDANO', 'DOGECOIN', 'RIPPLE','SOLANA']

#coins with date ordered in the strucutre: %d.%m.%y
dmy_coins = ['BINANCE','ETHEREUM','SHIBA_INU']
"""
for coin in coin_list:
    scrape_product_tweets(coin)

print("Product Set completely scraped")
"""
import os
import pandas as pd
from functions import date_to_epoch_intervall

for coin in dmy_coins:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'keyword_set_'+coin+".csv"
    keyword_df = pd.read_csv(os.path.join(my_path, my_file))
    keyword_df = keyword_df.head(100)
    for n,row in keyword_df.iterrows():
        print(row['date'])
        date_of_keyword = date_to_epoch_intervall(row['date'])
        print(date_of_keyword)
