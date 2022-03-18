df = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/1.Twitter_Scraping/ticker_set_#BTC.csv')
df = df.head(10)
text = df.iloc[9].content

print(text)

#version 1
tweet = text
# Remove urls
tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
# Remove user @ references and '#'
tweet = re.sub(r'\@\w+|\#','', tweet)
# Remove consequtive question marks
tweet = re.sub('[?]+[?]', ' ', tweet)
# Remove &amp - is HTML code for hyperlink
tweet = re.sub(r'\&amp;','&', tweet)
#Replace emoji with text
tweet = emoji.demojize(text, language="en", delimiters=(" ", " "))
print(tweet)
print("---------------")
#version2
text_2 = preprocess_tweet(text,lang="en")
print(text_2)
