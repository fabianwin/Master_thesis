from functions import scrape_product_tweets, scrape_google_trendwords, scrape_google_trendwords_year
import os
import pandas as pd

#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
#for product sets search for written out terms, than iterate through query findings
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA INU']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

"""
for coin in coin_list:
    scrape_google_trendwords(coin)

print("Product Set completely scraped")
"""



#scrape_google_trendwords("CARDANO")
#scrape_product_tweets("BTC")
scrape_google_trendwords_year("CARDANO",2017)

"""
#-----------Merge Datasets together
dates = ['2021', '2020', '2019', '2018', '2017']

def merge_Sets(keyword):
    pdList = []
    for year in dates:
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/Yearly_datasets/keywords_sets')
        my_file = 'keyword_set_'+keyword+"_"+year+".csv"
        df = pd.read_csv(os.path.join(my_path, my_file))
        pdList.append(df)

    entire_twitter_df = pd.concat(pdList)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping')
    my_file = 'keyword_set_'+keyword+".csv"
    entire_twitter_df.to_csv(os.path.join(my_path, my_file))


merge_Sets("DOGECOIN")
"""
