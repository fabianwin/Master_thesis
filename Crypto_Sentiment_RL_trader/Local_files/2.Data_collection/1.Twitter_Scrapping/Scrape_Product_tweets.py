from functions import scrape_product_tweets, scrape_google_trendwords

#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
classic_coins = ['BTC', 'ETH']
venture_capital_backed_coins = ['BNB', 'ADA', 'XRP']
community_driven_coins = ['DOGE', 'SHIB']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

"""
for coin in coin_list:
    scrape_product_tweets(coin)

print("Product Set completely scraped")
"""

#for testing
scrape_google_trendwords("SOL")
#scrape_product_tweets("ETH")
#scrape_product_tweets("BTC")
