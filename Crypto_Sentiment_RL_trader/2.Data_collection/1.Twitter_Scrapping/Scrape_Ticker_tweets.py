#Construct the Ticker Set for TSLA
from functions import get_ticker_tweets


#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
classic_coins = ['#BTC', '#ETH']
venture_capital_backed_coins = ['#SOL', '#ADA', '#XRP']
community_driven_coins = ['#DOGE', '#SHIB']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

"""
for coin in coin_list:
    get_ticker_tweets(coin)

print("Ticker Set completely scraped")
"""
for coin in community_driven_coins:
    get_ticker_tweets(coin)

print("Ticker Set completely scraped")


#for testing
get_ticker_tweets("#DOGE")
