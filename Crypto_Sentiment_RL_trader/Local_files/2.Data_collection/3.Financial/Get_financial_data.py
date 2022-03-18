from functions import get_intraday_data, get_daily_data

coin_list = ['BITCOIN', 'ETHEREUM','BINANCE', 'CARDANO', 'RIPPLE','DOGECOIN', 'SHIBA INU']

for coin in coin_list:
    get_intraday_data(coin)
    get_daily_data(coin)

print("Intraday financial data completely scraped")
