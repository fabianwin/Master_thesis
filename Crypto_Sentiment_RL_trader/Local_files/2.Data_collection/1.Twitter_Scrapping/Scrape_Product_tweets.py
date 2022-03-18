from functions import scrape_news_tweets

coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list:
    scrape_news_tweets(coin)

print("news Set completely scraped")
