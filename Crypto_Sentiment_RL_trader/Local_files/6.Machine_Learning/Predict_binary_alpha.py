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
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


########Functions########
def add_alpha_boolean(coin_df, index_df):
    index_df = pd.merge(index_df, coin_df.loc[:, ["date",'ROC_2']], how='left',  left_on="Date", right_on="date")
    index_df = index_df.drop(['date'], axis=1)
    index_df['alpha_return'] = index_df.ROC_2 - index_df.ROC_2_Index
    index_df['alpha_return_bool'] = np.sign(index_df.alpha_return)
    index_df['alpha_return_bool'].replace({-1: False, 1: True}, inplace=True)

    return index_df
#----------------------------
def finance_ROC(feature_df, n=2):
    """
    - Returns: Feature_set with an additional column describing the rate of change for each sentiment calculaiton method
    """
    roc_df = feature_df[['Close','Date']]
    roc_df.set_index('Date', inplace=True)
    Shift = roc_df.shift(n - 1)
    Diff = roc_df.diff(n - 1)
    ROC = pd.DataFrame(((Diff / Shift) * 100))
    new_column_name = 'ROC_'+str(n)
    ROC.rename({'Close': new_column_name}, axis=1, inplace=True)
    feature_df = pd.merge(feature_df, ROC, how='left',  left_on="Date", right_index=True)

    return feature_df
#----------------------------
def LogReg_Pred(feature_list, coin, set, feature_df, predict_return_df):
    #create df with all relevant features for this prediction
    column_list = feature_list
    column_list.append('positive_return')
    df = feature_df.loc[:,column_list]
    df.dropna(axis=0, how='any',inplace=True)
    feature_list.remove('positive_return')
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:,feature_list], df.loc[:,'positive_return'], test_size=0.3,random_state=42)

    #GridSearch
    clf = LogisticRegression()
    param_grid = {'penalty': ['l1'],
               'C':[0.001,.009,0.01,.09,1,5,10,25],
               'solver':['liblinear']}

    grid_clf_acc = GridSearchCV(clf, param_grid = param_grid,scoring = 'accuracy')
    grid_clf_acc.fit(X_train, y_train)
    grid_predictions = grid_clf_acc.predict(X_test)

    #Predict values based on new parameters
    y_pred_acc = grid_clf_acc.predict(X_test)

    #add metrcis to df
    new_row = {'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Logistic Regression",'Features':feature_list,'Accuracy_Score':accuracy_score(y_test,y_pred_acc), 'Precision_Score':precision_score(y_test,y_pred_acc), 'Recall_Score':recall_score(y_test,y_pred_acc), 'F1_Score':f1_score(y_test,y_pred_acc),'Best_Parameters':grid_clf_acc.best_params_}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

    return predict_return_df
#----------------------------
def SVM_Pred(feature_list, coin, set, feature_df,predict_return_df):
    #create df with all relevant features for this prediction
    column_list = feature_list
    column_list.append('positive_return')
    df = feature_df.loc[:,column_list]
    df.dropna(axis=0, how='any',inplace=True)
    feature_list.remove('positive_return')
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:,feature_list], df.loc[:,'positive_return'], test_size=0.3,random_state=42)

    svc = svm.SVC()
    #find the suitable parameters  for a rbf kernel
    param_grid = {'C': [0.35,0.4,0.45],
                  'gamma':['scale', 'auto'],
                  'kernel': ['rbf']}
    grid = GridSearchCV(svc, param_grid, refit = True, verbose = 3,n_jobs=-1)
    grid.fit(X_train, y_train)
    grid = grid.predict(X_test)
    print(grid)
    print(grid.best_params_)

    #Predict values based on new parameters
    y_pred_acc = grid.predict(X_test)
    print(y_pred_acc)

    #add metrcis to df
    new_row = {'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Logistic Regression",'Features':feature_list,'Accuracy_Score':accuracy_score(y_test,y_pred_acc), 'Precision_Score':precision_score(y_test,y_pred_acc), 'Recall_Score':recall_score(y_test,y_pred_acc), 'F1_Score':f1_score(y_test,y_pred_acc),'Best_Parameters':grid.best_params_}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

    perm_importance = permutation_importance(svc, X_train, y_train)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()

    return predict_return_df
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
########Main##########
#load BITX Index Data
my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/2.Data_collection/4.Financial_data/Daily_Data')
my_file = "BITX_index_fund.csv"
index_data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=["Exchange Date"], dayfirst=True)
index_data_df = index_data_df.rename(columns={"Exchange Date":"Date"})
index_data_df = finance_ROC(index_data_df)
index_data_df = index_data_df.rename(columns={"ROC_2":"ROC_2_Index"})

coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
sets=["ticker", "product"]

coins=['BTC']
sets=["ticker"]

predict_return = pd.DataFrame([], columns=['Coin','Set_description','supervised ML algorithm type','Features','Accuracy_Score', 'Precision_Score', 'Recall_Score', 'F1_Score'])
for coin in coins:
    for set in sets:
        #load feature dataset
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
        my_file = 'complete_feature_set_'+coin+".csv"
        my_scaled_file = 'scaled_complete_feature_set_'+coin+".csv"
        date_cols = ["date"]
        data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
        scaled_data_df = pd.read_csv(os.path.join(my_path, my_scaled_file), parse_dates=date_cols, dayfirst=True)
        if coin == 'BNB':
            data_df['Price (Close)'] = data_df['Price (High)']
            scaled_data_df['Price (Close)'] = scaled_data_df['Price (High)']

        #create Y column by creating the alpha signal
        index_df = add_alpha_boolean(data_df, index_data_df)
        data_df = pd.merge(data_df, index_df.loc[:, ["Date",'alpha_return','alpha_return_bool']], how='left',  left_on="date", right_on="Date")
        data_df = data_df.drop(['Date'], axis=1)


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
        predict_return = SVM_Pred(feature_list, coin, set,data_df, predict_return)

"""
        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
        predict_return = LogReg_Pred(feature_list, coin, set, scaled_data_df, predict_return)
        #---------------------------------

        #run with sentiment features only
        feature_list_appendable = ["_number_of_tweets", "_average_number_of_likes", "_average_number_of_retweets", "_average_number_of_followers", "_finiteautomata_sentiment","_finiteautomata_sentiment_expectation_value_volatility"]
        feature_list = [set + item for item in feature_list_appendable]
        if set == "ticker":
            feature_list.extend(("ROC_2_ticker_finiteautomata_sentiment","Momentum_14_ticker_finiteautomata_sentiment"))
        else:
            feature_list.extend(("ROC_2_product_finiteautomata_sentiment","Momentum_14_product_finiteautomata_sentiment"))

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
        predict_return = LogReg_Pred(feature_list, coin, set, scaled_data_df, predict_return)
        #---------------------------------

        #run with finance features only
        feature_list = ["Real Volume","Circulating Marketcap", "Sharpe Ratio", "Volatility", "MOM_14","RSI_14","pos_conf","neg_conf"]

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
        predict_return = LogReg_Pred(feature_list, coin, set, scaled_data_df, predict_return)
        #---------------------------------

        #run with network features only
        feature_list = ["Adjusted NVT","Adjusted RVT", "Deposits on Exchanges", "Withdrawals from Exchanges", "Average Transaction Fees", "Adjusted Transaction Volume", "Average Transfer Value", "Active Supply", "Miner Supply", "Miner Revenue per Hash per Second", "Addresses Count", "Active Addresses Count", "Addresses with balance greater than $1"]

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
        predict_return = LogReg_Pred(feature_list, coin, set, scaled_data_df, predict_return)
        #---------------------------------


predict_return.to_csv(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/6.Machine_Learning/return_predictions.csv', index = False)
"""