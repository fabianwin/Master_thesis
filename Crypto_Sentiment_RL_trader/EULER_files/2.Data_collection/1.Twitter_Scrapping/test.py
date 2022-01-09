import pandas as pd

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

print(df)

df.to_csv(r'test.csv')#SHIB_2021.csv
