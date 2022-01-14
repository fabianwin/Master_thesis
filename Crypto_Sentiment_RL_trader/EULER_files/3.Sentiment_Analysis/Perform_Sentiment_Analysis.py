import pandas as pd
import numpy as np
from functions import get_stanford_sentiment, get_textblob_sentiment, get_flair_sentiment, get_finiteautomata_sentiment, get_cardiffnlp_sentiment
import time

def perform_sentiment_analysis(df):
    toc_0 = time.perf_counter()
    #df = get_stanford_sentiment(df)
    toc_1 = time.perf_counter()
    print(f"Performed this function in {toc_1 - toc_0:0.4f} seconds")

    df = get_textblob_sentiment(df)
    toc_2 = time.perf_counter()
    print(f"Performed this function in {toc_2 - toc_1:0.4f} seconds")

    df = get_flair_sentiment(df)
    toc_3 = time.perf_counter()
    print(f"Performed this function in {toc_3 - toc_2:0.4f} seconds")

    #df = get_finiteautomata_sentiment(df)
    toc_4 = time.perf_counter()
    print(f"Performed this function in {toc_4 - toc_3:0.4f} seconds")

    #df = get_cardiffnlp_sentiment(df)
    toc_5 = time.perf_counter()
    print(f"Performed this function in {toc_5 - toc_4:0.4f} seconds")

    print(f"total time {toc_5 - toc_0:0.4f} seconds")
    return df


#load dataframe
btc_df = pd.read_csv(r'ticker_set_#BTC.csv')
btc_df = btc_df.head(100)
print(btc_df)

btc_df = perform_sentiment_analysis(btc_df)
btc_df.to_csv(r'ticker_set_#BTC_sentiment.csv', index = False)
