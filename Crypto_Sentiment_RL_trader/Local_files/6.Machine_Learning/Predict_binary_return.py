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
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno

processors = 10

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
def LogReg_Pred(feature_list, coin, set, feature_name, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    df_not_imputed.dropna(subset=['positive_return'], how="any", inplace=True)
    feature_list.remove('positive_return')
    train, test = train_test_split(df_not_imputed, test_size=0.3, random_state=42)
    X_train = train.loc[:,feature_list]
    y_train = pd.DataFrame(train.loc[:,'positive_return'])

    #Imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(imputed, columns=X_train.columns)

    #build a pipeline to find the best overall combination
    #define a standard scaler to normalize inputs
    std_slc = StandardScaler()
    #create a classifier regularization
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    logistic_Reg = LogisticRegression(max_iter=10000)

    #create actual pipeline
    pipe = Pipeline([
        ('scaler', std_slc),
        ('selector', "passthrough"),
        ('classifier', logistic_Reg)
    ])

    N_FEATURES_OPTIONS = list(range(1,len(feature_list)+1,1))
    C_OPTIONS = np.logspace(-4,4,10)
    reducer_labels = ['PCA']

    param_grid = [
        {
            'selector': [PCA(iterated_power=7)],
            'selector__n_components': N_FEATURES_OPTIONS,
            'classifier__C': C_OPTIONS,
            'classifier__solver':['lbfgs','liblinear']
        }
    ]

    #iterate over different imputation methods and scoring methods
    # tests have shown that iterative imputer is good enough and we want to optimize for accuracy.
    print("# Tuning hyper-parameters for %s" % coin)
    print()
    search = GridSearchCV(pipe, cv=5, scoring="accuracy", param_grid = param_grid, n_jobs=processors)
    search.fit(X_train, y_train.values.ravel())
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = search.cv_results_["mean_test_score"]
    stds = search.cv_results_["std_test_score"]

    print("Impute the test set")
    X_test = test.loc[:,feature_list]
    y_test = pd.DataFrame(test.loc[:,'positive_return'])
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_test)
    X_test = pd.DataFrame(imputed, columns=X_test.columns)

    print("Make the prediction")
    y_true, y_pred = y_test, search.predict(X_test)
    print(classification_report(y_true, y_pred))

    new_row = {'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Logistic Regression",'Features':feature_name,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

    return predict_return_df
#----------------------------
def KNN_Pred(feature_list, coin, set, feature_name, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    df_not_imputed.dropna(subset=['positive_return'], how="any", inplace=True)
    feature_list.remove('positive_return')

    train, test = train_test_split(df_not_imputed, test_size=0.3, random_state=42)
    X_train = train.loc[:,feature_list]
    y_train = pd.DataFrame(train.loc[:,'positive_return'])

    #imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(imputed, columns=X_train.columns)

    #build a pipeline to find the best overall combination
    #define a standard scaler to normalize inputs
    std_slc = StandardScaler()
    #create a classifier regularization
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    KNN = KNeighborsClassifier()

    #create actual pipeline
    pipe = Pipeline([
            ('scaler', std_slc),
            ('selector', "passthrough"),
            ('classifier', KNN)
    ])

    param_grid = [
        {
            'selector': [PCA()],
            'selector__n_components': list(range(1,len(feature_list)+1,1)),
            'classifier__n_neighbors': list(range(1, 150)),
            'classifier__p': [1,2],
        }
    ]

    #iterate over different imputation methods
    # tests have shown that iterative imputer is good enough and we want to optimize for accuracy.
    print("# Tuning hyper-parameters for %s" % coin)
    print()
    search = GridSearchCV(pipe, cv=5, scoring="accuracy", param_grid = param_grid, n_jobs=processors)
    search.fit(X_train, y_train.values.ravel())
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()

    print("Impute the test set")
    X_test = test.loc[:,feature_list]
    y_test = pd.DataFrame(test.loc[:,'positive_return'])
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_test)
    X_test = pd.DataFrame(imputed, columns=X_test.columns)

    print("Make the prediction")
    y_true, y_pred = y_test, search.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    new_row = {'Coin':coin,'Set_description': set,'supervised ML algorithm type':"K nearest Neighbour",'Features':feature_name,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

    return predict_return_df
#----------------------------
def SVM_Pred(feature_list, coin, set, feature_name, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    feature_list.remove('positive_return')
    train, test = train_test_split(df_not_imputed, test_size=0.3, random_state=42)
    X_train = train.loc[:,feature_list]
    y_train = pd.DataFrame(train.loc[:,'positive_return'])

    #Imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(imputed, columns=X_train.columns)

    #build a pipeline to find the best overall combination
    #define a standard scaler to normalize inputs
    std_slc = StandardScaler()
    #create a classifier regularization
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    SVC = svm.SVC()

    #create actual pipeline
    pipe = Pipeline([
        ('scaler', std_slc),
        ('selector', "passthrough"),
        ('classifier', SVC)
    ])

    N_FEATURES_OPTIONS = list(range(1,len(feature_list)+1,1))
    C_OPTIONS =  [0.1, 0.35,0.4,0.45, 1, 10, 100, 1000, 10000, 100000, 1000000]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    reducer_labels = ['PCA']

    param_grid = [
        {
            'selector': [PCA()],
            'selector__n_components': N_FEATURES_OPTIONS,
            'classifier__C': C_OPTIONS,
            'classifier__gamma': gamma,
            'classifier__kernel':['rbf']
        }
    ]

    #iterate over different imputation methods and scoring methods
    # tests have shown that iterative imputer is good enough and we want to optimize for accuracy.
    print("# Tuning hyper-parameters for %s" % coin)
    print()
    search = GridSearchCV(pipe, cv=5, scoring="accuracy", param_grid = param_grid, n_jobs=processors)
    search.fit(X_train, y_train.values.ravel())
    print("Best parameters set found on development set:")
    print()
    print(search.best_params_)
    print()

    print("Impute the test set")
    X_test = test.loc[:,feature_list]
    y_test = pd.DataFrame(test.loc[:,'positive_return'])
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(X_test)
    X_test = pd.DataFrame(imputed, columns=X_test.columns)

    print("Make the prediction")
    y_true, y_pred = y_test, search.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    new_row = {'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Support Vector Machine",'Features':feature_name,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
    predict_return_df= predict_return_df.append(new_row, ignore_index=True)

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
coins=['ADA','BNB','BTC','DOGE','ETH', 'XRP']
sets=["ticker", "news"]

predict_return = pd.DataFrame([], columns=['Coin','Set_description','supervised ML algorithm type','Features','Accuracy_Score', 'Precision_Score', 'Recall_Score', 'F1_Score'])
for afunc in (LogReg_Pred, KNN_Pred, SVM_Pred):
    for coin in coins:
        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/4.Feature_Engineering/Daily_trading')
        my_file = 'complete_feature_set_'+coin+".csv"
        date_cols = ["date"]
        data_df = pd.read_csv(os.path.join(my_path, my_file), parse_dates=date_cols, dayfirst=True)
        data_df = add_return_boolean(data_df)

        #run with features from Xiao/Chen Paper (8 features)
        set = "ticker"
        feature_name = "Xiao/Chen Paper"
        feature_list = ["ticker_number_of_tweets", "ticker_finiteautomata_sentiment", "ticker_finiteautomata_sentiment_expectation_value_volatility", "ticker_average_number_of_followers","Momentum_14_ticker_finiteautomata_sentiment","MOM_14","Volatility","RSI_14"]
        predict_return = afunc(feature_list, coin, set, feature_name, data_df, predict_return)

        #run with ticker sentiment features only (8 features)
        set = "ticker"
        feature_name = "Sentiment features"
        feature_list_1 = ["ticker_number_of_tweets", "ticker_average_number_of_likes", "ticker_average_number_of_retweets", "ticker_average_number_of_followers", "ticker_finiteautomata_sentiment","ticker_finiteautomata_sentiment_expectation_value_volatility","ROC_2_ticker_finiteautomata_sentiment","Momentum_14_ticker_finiteautomata_sentiment"]
        predict_return = afunc(feature_list_1, coin, set, feature_name, data_df, predict_return)

        #run with news sentiment features only (8 features)
        set = "news"
        feature_name = "Sentiment features"
        feature_list_2 = ["news_number_of_tweets", "news_average_number_of_likes", "news_average_number_of_retweets", "news_average_number_of_followers", "news_finiteautomata_sentiment","news_finiteautomata_sentiment_expectation_value_volatility","ROC_2_news_finiteautomata_sentiment","Momentum_14_news_finiteautomata_sentiment"]
        predict_return = afunc(feature_list_2, coin, set, feature_name, data_df, predict_return)

        # run with finance features only (8 features)
        set = "NaN"
        feature_name = "Finance features"
        feature_list_3 = ["Circulating Marketcap", "Adjusted NVT","Adjusted RVT", "Sharpe Ratio", "Volatility", "MOM_14","RSI_14","pos_conf","neg_conf"]
        predict_return = afunc(feature_list_3, coin, set, feature_name, data_df, predict_return)

        # run with network features only (8 features)
        set = "NaN"
        feature_name = "Network features"
        feature_list_4 = ["Median Transaction Fees", "Adjusted Transaction Volume", "Median Transfer Value","Transactions Count", "Active Supply", "Addresses Count", "Active Addresses Count", "Active Addresses Count (Received)","Active Addresses Count (Sent)","Addresses with balance greater than $1"]
        predict_return = afunc(feature_list_4, coin, set, feature_name, data_df, predict_return)

        #run with all features (32 features)
        set = "NaN"
        feature_name = "All features"
        feature_list_big = feature_list_1 + feature_list_2 + feature_list_3 + feature_list_4
        predict_return = afunc(feature_list_big, coin, set, feature_name, data_df, predict_return)

        df = predict_return.loc[predict_return['Coin']==coin]
        #df.loc[:,['Accuracy_Score','Precision_Score','Recall_Score','F1_Score']]
        #df["Accuracy_Score"].mean()
        new_row = {'Coin':coin,'Set_description': "Average",'supervised ML algorithm type':str(afunc),'Features':" ",'Accuracy_Score':df["Accuracy_Score"].mean(), 'Precision_Score':df["Precision_Score"].mean(), 'Recall_Score':df["Recall_Score"].mean(), 'F1_Score':df["F1_Score"].mean(),'Best_Parameters': " "}
        predict_return= predict_return.append(new_row, ignore_index=True)

        my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/6.Machine_Learning')
        my_file = str(afunc)+" _new_predictions.csv"
        predict_return.to_csv(os.path.join(my_path, my_file))
