#########Notes############
"""
Predict if the return of the next day will be positive or negative or positive with relevant sentiment features (from 6.0) with a logistic Regression

1. use the relevant sentiment features:
2. find which sentiment features increases the accuracy
3. find the accuracy for predicting return with baseline and sentiment features

Finding:
We are only select feature which improves the accuracy score (same as in paper). In our case we would use sentiment volatility, sentiment momentum, stanford sentiment as a sentimental features. This finding holds true even when we hold tuning parameters constant.
"""
########Libraries########
from sklearn.model_selection import train_test_split
from sklearn import svm
from pandas import DataFrame
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

########Functions########
def add_return_boolean(df):
    for i, row in df.iterrows():
        if row['next day return'] >= 0:
            sig = 1
        else:
            sig=0
        df.at[i, 'positive return'] = sig
    return df


def metrics_for_features(feature_list, feature_df, predict_return_df):
    print("start new metric with", list(feature_df.iloc[:,feature_list].columns.values))
    X_train, X_test, y_train, y_test = train_test_split(feature_df.iloc[:,feature_list], feature_df.iloc[:,12], test_size=0.3,random_state=42)

    #GridSearch
    clf = LogisticRegression()
    param_grid = {'penalty': ['l1'],
               'C':[0.001,.009,0.01,.09,1,5,10,25],
               'solver':['liblinear']}

    grid_clf_acc = GridSearchCV(clf, param_grid = param_grid,scoring = 'accuracy')
    grid_clf_acc.fit(X_train, y_train)
    grid_predictions = grid_clf_acc.predict(X_test)
    print("Best Paramteres for this configuration are:", grid_clf_acc.best_params_)
    print("")

    #Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)
    #add metrcis to df
    new_row = {'Features':list(feature_df.iloc[:,feature_list].columns.values),'Accuracy Score':accuracy_score(y_test,y_pred_acc), 'Precision Score':precision_score(y_test,y_pred_acc), 'Recall Score':recall_score(y_test,y_pred_acc), 'F1 Score ':f1_score(y_test,y_pred_acc)}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

    return predict_return_df
def normalize(df):
    scaler=StandardScaler()
    for column in df.columns[1:12]:
        scaled_data = scaler.fit_transform(df[[column]])
        df[column] = scaled_data
    return df

    
########Main##########
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Ticker_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Ticker_TSLA.csv')
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.dropna(axis=0, how="any")
Feature_set_Ticker_TSLA = add_return_boolean(Feature_set_Ticker_TSLA)
Feature_set_Ticker_TSLA = Feature_set_Ticker_TSLA.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Ticker_TSLA = normalize(Feature_set_Ticker_TSLA)

predict_return = pd.DataFrame([], columns=['Features','Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score '])
predict_return = metrics_for_features([1,2,3,4], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility']
predict_return = metrics_for_features([1,2,3,4,5], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility']
predict_return = metrics_for_features([1,2,3,4,5,6], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum']
predict_return = metrics_for_features([1,2,3,4,5,6,7], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets']
predict_return = metrics_for_features([1,2,3,4,5,6,8], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Stanford Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,9], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'TextBlob Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,10], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Flair Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,11], Feature_set_Ticker_TSLA, predict_return) # ['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8,9,10,11], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment', 'TextBlob Sentiment', 'Flair Sentiment', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8], Feature_set_Ticker_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment']

#Same procedure for product set
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Product_TSLA = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Product_TSLA.csv')
Feature_set_Product_TSLA = Feature_set_Product_TSLA.dropna(axis=0, how="any")
Feature_set_Product_TSLA = add_return_boolean(Feature_set_Product_TSLA)
Feature_set_Product_TSLA =  Feature_set_Product_TSLA.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Product_TSLA = normalize(Feature_set_Product_TSLA)

new_row = {'Features': "TESLA product set"}
predict_return= predict_return.append(new_row, ignore_index=True)
predict_return = metrics_for_features([1,2,3,4], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility']
predict_return = metrics_for_features([1,2,3,4,5], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility']
predict_return = metrics_for_features([1,2,3,4,5,6], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum']
predict_return = metrics_for_features([1,2,3,4,5,6,7], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets']
predict_return = metrics_for_features([1,2,3,4,5,6,8], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Stanford Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,9], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'TextBlob Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,10], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Flair Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,11], Feature_set_Product_TSLA, predict_return) # ['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8,9,10,11], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment', 'TextBlob Sentiment', 'Flair Sentiment', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8], Feature_set_Product_TSLA, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment']

#Same procedure for product set with Imputed features
#prepare the dataset with technical features only. Using self-choosen features --> same day return instead of previou's days return
Feature_set_Product_TSLA_imp = pd.read_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Input/1. Data Daily/Feature_set_Product_TSLA_imp.csv')
Feature_set_Product_TSLA_imp = Feature_set_Product_TSLA_imp.dropna(axis=0, how="any")
Feature_set_Product_TSLA_imp = add_return_boolean(Feature_set_Product_TSLA_imp)
Feature_set_Product_TSLA_imp =  Feature_set_Product_TSLA_imp.drop(columns=['next day return', "previous day's return", 'open', 'close'])
Feature_set_Product_TSLA_imp = normalize(Feature_set_Product_TSLA_imp)

new_row = {'Features': "TESLA product set with imputed features"}
predict_return= predict_return.append(new_row, ignore_index=True)

predict_return = metrics_for_features([1,2,3,4], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility']
predict_return = metrics_for_features([1,2,3,4,5], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility']
predict_return = metrics_for_features([1,2,3,4,5,6], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum']
predict_return = metrics_for_features([1,2,3,4,5,6,7], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets']
predict_return = metrics_for_features([1,2,3,4,5,6,8], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Stanford Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,9], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'TextBlob Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,10], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'Flair Sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,11], Feature_set_Product_TSLA_imp, predict_return) # ['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8,9,10,11], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment', 'TextBlob Sentiment', 'Flair Sentiment', 'normalized average sentiment']
predict_return = metrics_for_features([1,2,3,4,5,6,7,8], Feature_set_Product_TSLA_imp, predict_return) #['same day return', 'volume', 'price momentum', 'price volatility', 'sentiment volatility', 'sentiment momentum', 'number_of_tweets', 'Stanford Sentiment']

pd.set_option('display.max_columns', None)
print(predict_return)
predict_return.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Trading the twitter sentiment replica/Output/1. Output Daily/return_predictions_for_different_feature_sets(same_day_return).csv', index = False)
