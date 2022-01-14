from functions import scrape_product_tweets, scrape_google_trendwords, scrape_google_trendwords_year

#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
#for product sets search for written out terms, than iterate through query findings
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA INU']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

"""
for coin in coin_list:
    scrape_product_tweets(coin)

print("Product Set completely scraped")
"""

#for testing
"""
coin_list = venture_capital_backed_coins + community_driven_coins
for coin in coin_list:
    scrape_google_trendwords(coin)
"""


scrape_google_trendwords("ETHEREUM")
#scrape_product_tweets("BTC")
#scrape_google_trendwords("BINANCE",2017)
