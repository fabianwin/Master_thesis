import pandas as pd
import numpy as np
from functions import get_stanford_sentiment, get_textblob_sentiment, get_flair_sentiment, get_finiteautomata_sentiment, get_cardiffnlp_sentiment
import time

def perform_sentiment_analysis(df):
    toc_0 = time.perf_counter()
    df = get_stanford_sentiment(df)
    toc_1 = time.perf_counter()
    print(f"Performed this function in {toc_1 - toc_0:0.4f} seconds")

    df = get_textblob_sentiment(df)
    toc_2 = time.perf_counter()
    print(f"Performed this function in {toc_2 - toc_1:0.4f} seconds")

    df = get_flair_sentiment(df)
    toc_3 = time.perf_counter()
    print(f"Performed this function in {toc_3 - toc_2:0.4f} seconds")

    df = get_finiteautomata_sentiment(df)
    toc_4 = time.perf_counter()
    print(f"Performed this function in {toc_4 - toc_3:0.4f} seconds")

    df = get_cardiffnlp_sentiment(df)
    toc_5 = time.perf_counter()
    print(f"Performed this function in {toc_5 - toc_4:0.4f} seconds")

    print(f"total time {toc_5 - toc_0:0.4f} seconds")
    return df

coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN']
for coin in coin_list_news:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'news_set_preprocessed_'+coin+".csv"
    df = pd.read_csv(os.path.join(my_path, my_file))
    df = perform_sentiment_analysis(df)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/')
    my_file = 'news_set_sentiment_'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))

coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
for coin in coins:
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/2.Twitter_processing')
    my_file = 'ticker_set_preprocessed_#'+coin+".csv"
    df = pd.read_csv(os.path.join(my_path, my_file))
    df = perform_sentiment_analysis(df)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/3.Sentiment_Analysis/')
    my_file = 'ticker_set_sentiment_#'+coin+".csv"
    df.to_csv(os.path.join(my_path, my_file))
