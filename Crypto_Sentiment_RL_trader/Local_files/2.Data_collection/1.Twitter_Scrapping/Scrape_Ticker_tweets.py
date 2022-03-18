from functions import get_ticker_tweets

coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list:
    get_ticker_tweets(coin)

print("Ticker Set completely scraped")
