import pandas as pd

# initialize data of lists.
data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18]}

print("Hello EULER")

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv(r'test.csv',index=False)
