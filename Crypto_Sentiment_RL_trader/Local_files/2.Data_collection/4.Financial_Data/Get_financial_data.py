from functions import get_intraday_data, get_daily_data

classic_coins = ['BTC', 'ETH']
venture_capital_backed_coins = ['BNB','SOL', 'ADA', 'XRP']
community_driven_coins = ['DOGE', 'SHIB']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

for coin in coin_list:
    get_intraday_data(coin)
    get_daily_data(coin)

print("Intraday financial data completely scraped")
