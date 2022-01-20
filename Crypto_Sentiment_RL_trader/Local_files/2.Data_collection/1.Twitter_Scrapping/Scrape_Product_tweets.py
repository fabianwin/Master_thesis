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
