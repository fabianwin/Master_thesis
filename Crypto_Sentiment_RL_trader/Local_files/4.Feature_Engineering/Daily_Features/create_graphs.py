from functions import get_lppls_graphs

classic_coins = ['BTC', 'ETH']
venture_capital_backed_coins = ['BNB','ADA', 'XRP']
community_driven_coins = ['DOGE']
coin_list = classic_coins + venture_capital_backed_coins + community_driven_coins

for coin in coin_list:
    print(coin)
    get_lppls_graphs(coin)
