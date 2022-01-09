import pandas as pd

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

print(df)

df.to_csv(r'Crypto_Sentiment_RL_trader/EULER_files/2.Data_collection/1.Twitter_Scrapping/Data/ticker_sets/test.csv')#SHIB_2021.csv
