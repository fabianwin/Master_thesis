from functions import scrape_product_tweets


#define which coins we scrape twitter data, only look at coins in top20 marketgap (January 2022)
#for product sets search for written out terms, than iterate through query findings
classic_coins = ['BITCOIN', 'ETHEREUM']
venture_capital_backed_coins = ['BINANCE', 'CARDANO', 'RIPPLE']
community_driven_coins = ['DOGECOIN', 'SHIBA INU']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins


scrape_product_tweets('DOGECOIN')
