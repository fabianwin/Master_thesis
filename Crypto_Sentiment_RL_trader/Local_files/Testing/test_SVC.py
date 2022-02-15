#########Notes############
"""
Predict if the return of the next day will be positive or negative or positive #with relevant sentiment features (from 6.0) with a logistic Regression

1. use the relevant sentiment features:
2. find which sentiment features increases the accuracy
3. find the accuracy for predicting return with baseline and sentiment features

Finding:
We are only select feature which improves the accuracy score (same as in paper). In our case we would use sentiment volatility, sentiment momentum, stanford sentiment as a sentimental features. This finding holds true even when we hold tuning parameters constant.
"""
########Libraries########
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


########Functions########
def add_return_boolean(df):
    for i, row in df.iterrows():
        if row['ROC_2'] >= 0:
            sig = 1
        else:
            sig= 0
        df.at[i, 'positive_return'] = sig
    return df
#----------------------------
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns
#----------------------------
def SVM_Pred(feature_list, coin, set, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    df_not_imputed.dropna(axis=0, how='any',inplace=True)
    feature_list.remove('positive_return')

    X = df_not_imputed.loc[:,feature_list]
    Y = pd.DataFrame(df_not_imputed.loc[:,'positive_return'])
    X_miss,Y_miss = missing_values_table(X), missing_values_table(Y)
    print()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

    #Create a svm Classifier
    svc = svm.SVC() # Linear Kernel
    param_grid = {'C': [0.1, 0.35,0.4,0.45, 1, 10, 100, 1000, 10000, 100000, 1000000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ["rbf"]}

    grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1)

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    

    # print prediction results
    print(classification_report(y_test, y_pred))


    return predict_return_df
#----------------------------


########Main##########
#coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
#sets=["ticker", "product"]

coins=['BTC']
sets=["ticker"]

predict_return = pd.DataFrame([], columns=['Coin','Set_description','supervised ML algorithm type','Features','Accuracy_Score', 'Precision_Score', 'Recall_Score', 'F1_Score'])
for coin in coins:
    for set in sets:
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
        my_file = 'complete_feature_set_'+coin+".csv"
        my_scaled_file = 'scaled_complete_feature_set_'+coin+".csv"
        date_cols = ["date"]
        data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
        scaled_data_df = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)
        data_df = add_return_boolean(data_df)
        scaled_data_df['positive_return'] = data_df.positive_return

        feature_list_appendable = ["_number_of_tweets", "_finiteautomata_sentiment", "_finiteautomata_sentiment_expectation_value_volatility", "_average_number_of_followers", "_finiteautomata_sentiment"]
        feature_list = [set + item for item in feature_list_appendable]
        print(feature_list)
        print()
        predict_return = SVM_Pred(feature_list, coin, set, data_df, predict_return)




        """
        #miss_df = missing_values_table(data_df)
        #print(miss_df)
        #data_df.info(verbose=True)

        #run with features from Chen Paper
        feature_list_appendable = ["_number_of_tweets", "_finiteautomata_sentiment", "_finiteautomata_sentiment_expectation_value_volatility", "_average_number_of_followers", "_finiteautomata_sentiment"]
        feature_list = [set + item for item in feature_list_appendable]
        if set == "ticker":
            feature_list.append("Momentum_14_ticker_finiteautomata_sentiment")
        else:
            feature_list.append("Momentum_14_product_finiteautomata_sentiment")
        feature_list.extend(["Real Volume","MOM_14","Volatility","RSI_14"])
        #predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
        predict_return = SVM_Pred(feature_list, coin, set, data_df, predict_return)
        print("-------------------------------------")
        """


predict_return.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/6.Machine_Learning/return_SVM_predictions.csv', index = False)
