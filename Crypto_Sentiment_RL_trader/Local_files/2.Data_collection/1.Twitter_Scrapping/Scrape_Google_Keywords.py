from functions import scrape_google_trendwords

oin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list:
    scrape_google_trendwords(coin)

print("Google queries completely scraped")
