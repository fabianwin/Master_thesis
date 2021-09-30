import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pingouin as pg
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
sns.set(style='white', font_scale=1.2)
import os
import re

#----------------------------------
def draw_correlation_plot(feature_df,name_str ,sentiment_str):
    """
    - Parameters: twitter_df & feature_df (Both df), twitter has all the tweets info stored, features need to be extracted and appended to df
    - Returns: df_final, same shape as df but with the inputed features
    """
    row_name = "daily average "+sentiment_str+" score"

    #normalize data
    data=feature_df[row_name]
    data=(2*(data - np.min(data)) / (np.max(data) - np.min(data)))-1 #normalize between [-1,1]
    data= data*0.5 #normalize between [-0.5,0.5]

    #check correlation find informaiton for the variables here https://raphaelvallat.com/correlation.html
    print(pg.corr(x=data, y=feature_df["previous day's return"]))

    #draw plot
    g = sns.jointplot(data=feature_df, x=data, y="previous day's return")
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Graphs') # Figures out the absolute path for you in case your working directory moves around.
    my_file = name_str+" "+sentiment_str+'.png'
    plt.savefig(os.path.join(my_path, my_file))
#----------------------------------
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")

#----------------------------------
def preprocess_tweet(tweet):
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#'
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove consequtive question marks
    tweet = re.sub('[?]+[?]', ' ', tweet)
    return tweet
#----------------------------------
def create_wordcloud(twitter_df,i,name_str, word_list):
    text = " ".join(review for review in twitter_df.iloc[:,i])
    text = preprocess_tweet(text)
    print ("There are {} words in the combination of all requirements.".format(len(text)))
    # Create stopword list:
    stopwords = set(STOPWORDS)
    #stopwords.update(["t", "m","u","p","y","TSLA"])
    stopwords.update(word_list)
    # Generate a word cloud image
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, max_words=80 ,background_color='white', collocations=False, stopwords = stopwords).generate(text)
    plot_cloud(wordcloud)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Graphs') # Figures out the absolute path for you in case your working directory moves around.
    my_file = name_str+'.png'
    wordcloud.to_file(os.path.join(my_path, my_file))


#----------------------------------

#Visualize different twitter datasets
ticker_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_TSLA.csv')
ticker_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/ticker_set_GM.csv')
product_set_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_TSLA.csv')
product_set_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/product_set_GM.csv')


create_wordcloud(ticker_set_TSLA,5, "ticker_TSLA_wordcloud", ["t", "m","u","p","y","TSLA","amp"])
create_wordcloud(ticker_set_GM,5, "ticker_GM_wordcloud", ["t", "m","u","p","y","GM","amp"])
create_wordcloud(product_set_TSLA,3, "product_TSLA_wordcloud", ["t", "m","u","p","y","TSLA","amp"])
create_wordcloud(product_set_GM,3, "product_GM_wordcloud", ["t", "m","u","p","y","GM","amp"])


'''
#Visualize different Sentiment Scores
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_GM = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/Feature_set_Ticker_GM.csv')

draw_correlation_plot(Feature_set_Ticker_TSLA,"ticker_TSLA", "Stanford_sentiment")
draw_correlation_plot(Feature_set_Ticker_TSLA,"ticker_TSLA",  "TextBlob_sentiment")
draw_correlation_plot(Feature_set_Ticker_TSLA,"ticker_TSLA",  "Flair_sentiment")

draw_correlation_plot(Feature_set_Ticker_GM,"ticker_GM", "Stanford_sentiment")
draw_correlation_plot(Feature_set_Ticker_GM,"ticker_GM",  "TextBlob_sentiment")
draw_correlation_plot(Feature_set_Ticker_GM,"ticker_GM",  "Flair_sentiment")
'''
