import pandas as pd

# initialize data of lists.
data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18]}

print("Hello world")

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv(r'test.csv',index=False)

#df = pd.read_csv(r'Master_thesis/Crypto_Sentiment_RL_trader/EULER_files/3.Sentiment Analysis/test.csv')
#print(df.head())
