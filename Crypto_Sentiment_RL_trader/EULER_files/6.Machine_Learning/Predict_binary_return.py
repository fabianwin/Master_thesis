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
import missingno as msno


processors = 8

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
def LogReg_Pred(feature_list, coin, set, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    feature_list.remove('positive_return')

    #display nullity
    """
    print(df_not_imputed.shape)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/5.Statistical_Analysis/Daily_trading/Missing_Data')
    msno.matrix(df_not_imputed,fontsize=5, figsize=(5, 5))
    my_file = coin+ "_matrix_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    msno.heatmap(df_not_imputed,fontsize=8, figsize=(5, 5))
    my_file = coin+ "_heatmap_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    msno.dendrogram(df_not_imputed,fontsize=8, figsize=(5, 5))
    my_file = coin+ "_dendrogram_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    """

    print("shape of total df",df_not_imputed.shape)
    print()
    train, test = train_test_split(df_not_imputed, test_size=0.3, random_state=42)
    print("train",train.shape)
    print("test",test.shape)
    #Imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(train)
    train = pd.DataFrame(imputed, columns=df_not_imputed.columns)
    print("Train NaN count = ",train.isna().sum().sum())
    X_train = train.loc[:,feature_list]
    y_train = pd.DataFrame(train.loc[:,'positive_return'])

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
    reducer_labels = ['PCA', 'KBest(f_classif, k="all")']

    param_grid = [
        {
            'selector': [PCA(iterated_power=7)],
            'selector__n_components': N_FEATURES_OPTIONS,
            'classifier__C': C_OPTIONS,
            'classifier__solver':['lbfgs','liblinear']
        },
        {
            'selector': [SelectKBest(chi2)],
            'selector__k': N_FEATURES_OPTIONS,
            'classifier__C': C_OPTIONS,
            'classifier__solver':['lbfgs','liblinear']
        },
    ]

    #iterate over different imputation methods and scoring methods
    # tests have shown that iterative imputer is good enough and we want to optimize for accuracy.
    dfs = [df_not_imputed]
    scores = ["accuracy"]
    for i, df in enumerate(dfs):
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            search = GridSearchCV(pipe, cv=5, scoring=score, param_grid = param_grid, n_jobs=processors)
            search.fit(X_train, y_train.values.ravel())
            print("Best parameters set found on development set:")
            print()
            print(search.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = search.cv_results_["mean_test_score"]
            stds = search.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, search.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()
            print("Impute the test set")
            imputer = IterativeImputer(random_state=42)
            imputed = imputer.fit_transform(test)
            test = pd.DataFrame(imputed, columns=df_not_imputed.columns)
            X_test = test.loc[:,feature_list]
            y_test = pd.DataFrame(test.loc[:,'positive_return'])
            print()

            print("Make the prediction")
            y_true, y_pred = y_test, search.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            importance = search.best_estimator_[2].coef_[0]
            print(search.best_params_)
            for i,v in enumerate(importance):
            	print('Feature: %0d, Score: %.5f' % (i,v))
            result_df = pd.DataFrame.from_dict(search.cv_results_, orient='columns')
            new_row = {'columns':list(result_df.columns),'score': score,'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Logistic Regression",'Features':feature_list,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
            predict_return_df= predict_return_df.append(new_row, ignore_index=True)
            predict_return.to_csv(r'return_logreg_predictions.csv', index = False)

    return predict_return_df
#----------------------------
def SVM_Pred(feature_list, coin, set, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    feature_list.remove('positive_return')

    #display nullity
    """
    print(df_not_imputed.shape)
    my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/5.Statistical_Analysis/Daily_trading/Missing_Data')
    msno.matrix(df_not_imputed,fontsize=5, figsize=(5, 5))
    my_file = coin+ "_matrix_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    msno.heatmap(df_not_imputed,fontsize=8, figsize=(5, 5))
    my_file = coin+ "_heatmap_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    msno.dendrogram(df_not_imputed,fontsize=8, figsize=(5, 5))
    my_file = coin+ "_dendrogram_nullity"
    plt.savefig(os.path.join(my_path, my_file), format='pdf', bbox_inches='tight')
    """

    print("shape of total df",df_not_imputed.shape)
    print()
    train, test = train_test_split(df_not_imputed, test_size=0.3, random_state=42)
    print("train",train.shape)
    print("test",test.shape)
    #Imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(train)
    train = pd.DataFrame(imputed, columns=df_not_imputed.columns)
    print("Train NaN count = ",train.isna().sum().sum())
    X_train = train.loc[:,feature_list]
    y_train = pd.DataFrame(train.loc[:,'positive_return'])

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
    dfs = [df_not_imputed]
    scores = ["accuracy"]
    for i, df in enumerate(dfs):
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            search = GridSearchCV(pipe, cv=5, scoring=score, param_grid = param_grid, n_jobs=processors)
            search.fit(X_train, y_train.values.ravel())
            print("Best parameters set found on development set:")
            print()
            print(search.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = search.cv_results_["mean_test_score"]
            stds = search.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, search.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()
            print("Impute the test set")
            imputer = IterativeImputer(random_state=42)
            imputed = imputer.fit_transform(test)
            test = pd.DataFrame(imputed, columns=df_not_imputed.columns)
            X_test = test.loc[:,feature_list]
            y_test = pd.DataFrame(test.loc[:,'positive_return'])
            print()

            print("Make the prediction")
            y_true, y_pred = y_test, search.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            importance = search.best_estimator_[2].coef_[0]
            print(search.best_params_)
            for i,v in enumerate(importance):
            	print('Feature: %0d, Score: %.5f' % (i,v))
            result_df = pd.DataFrame.from_dict(search.cv_results_, orient='columns')
            new_row = {'columns':list(result_df.columns),'score': score,'Coin':coin,'Set_description': set,'supervised ML algorithm type':"Logistic Regression",'Features':feature_list,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
            predict_return_df= predict_return_df.append(new_row, ignore_index=True)
            predict_return.to_csv(r'return_SVM_predictions.csv', index = False)

    return predict_return_df
#----------------------------
def KNN_Pred(feature_list, coin, set, feature_df, predict_return_df):
    #create X and Y datasets
    feature_list.append('positive_return')
    df_not_imputed = feature_df.loc[:,feature_list]
    feature_list.remove('positive_return')

    #iterative imputer
    imputer = IterativeImputer(random_state=42)
    imputed = imputer.fit_transform(df_not_imputed)
    df_iterative_imputed = pd.DataFrame(imputed, columns=df_not_imputed.columns)

    #build a pipeline to find the best overall combination
    #define a standard scaler to normalize inputs
    std_slc = StandardScaler()
    #create a classifier regularization
    pca = PCA()
    # set the tolerance to a large value to make the example faster
    knn = KNeighborsClassifier()

    #create actual pipeline
    pipe = Pipeline([
            ('scaler', std_slc),
            ('selector', "passthrough"),
            ('classifier', SVC)
    ])


    N_FEATURES_OPTIONS = list(range(1,len(feature_list)+1,1))
    C_OPTIONS =  [0.1, 0.35,0.4,0.45, 1, 10, 100, 1000, 10000, 100000, 1000000]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    #reducer_labels = ['PCA', 'KBest(f_classif, k="all")']
    reducer_labels = ['PCA']

    param_grid = [
        {
            'selector': [PCA()],
            'selector__n_components': N_FEATURES_OPTIONS,
            'classifier__n_neighbors': list(range(1, 150)),
            'classifier__p': [1,2],
        }
    ]

    scores = ["accuracy"]
    #iterate over different imputation methods
    dfs = [df_iterative_imputed]
    for i, df in enumerate(dfs):
        print(df.shape)
        X = df.loc[:,feature_list]
        Y = pd.DataFrame(df.loc[:,'positive_return'])
        X_miss,Y_miss = missing_values_table(X), missing_values_table(Y)
        print()
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=42)

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            search = GridSearchCV(pipe, cv=5, scoring=score, param_grid = param_grid, n_jobs=processors)
            search.fit(X_train, y_train.values.ravel())

            print("Best parameters set found on development set:")
            print()
            print(search.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = search.cv_results_["mean_test_score"]
            stds = search.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, search.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, search.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

            """
            pyplot.bar([x for x in range(len(importance))], importance)
            pyplot.show()
            my_path = os.path.abspath(r'/Users/fabianwinkelmann/Library/Mobile Documents/com~apple~CloudDocs/Master Thesis/Code/Crypto_Sentiment_RL_trader/6.Machine_Learning')
            my_file = 'Feature:importance(%0d).png' % (i)
            plt.savefig(os.path.join(my_path, my_file))¨
            """

            result_df = pd.DataFrame.from_dict(search.cv_results_, orient='columns')
            new_row = {'columns':list(result_df.columns),'score': score,'Coin':coin,'Set_description': set,'supervised ML algorithm type':"SVM",'Features':feature_list,'Accuracy_Score':accuracy_score(y_true,y_pred), 'Precision_Score':precision_score(y_true,y_pred), 'Recall_Score':recall_score(y_true,y_pred), 'F1_Score':f1_score(y_true,y_pred),'Best_Parameters':search.best_params_}
            predict_return_df= predict_return_df.append(new_row, ignore_index=True)
            predict_return.to_csv(r'return_KNN_predictions.csv', index = False)

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
sets=["ticker", "product"]

predict_return = pd.DataFrame([], columns=['Coin','Set_description','supervised ML algorithm type','Features','Accuracy_Score', 'Precision_Score', 'Recall_Score', 'F1_Score'])
for afunc in (LogReg_Pred, SVM_Pred, KNN_Pred):
    for coin in coins:
        for set in sets:
            print(coin)
            my_file = 'Daily_trading/complete_feature_set_'+coin+".csv"
            my_scaled_file = 'Daily_trading/scaled_complete_feature_set_'+coin+".csv"
            date_cols = ["date"]
            data_df = pd.read_csv(my_file, parse_dates=date_cols, dayfirst=True)
            scaled_data_df = pd.read_csv(my_scaled_file, parse_dates=date_cols, dayfirst=True)
            data_df = add_return_boolean(data_df)
            scaled_data_df['positive_return'] = data_df.positive_return

            #run with features from Chen Paper (9 features)
            feature_list_appendable = ["_number_of_tweets", "_finiteautomata_sentiment", "_finiteautomata_sentiment_expectation_value_volatility", "_average_number_of_followers", "_finiteautomata_sentiment"]
            feature_list = [set + item for item in feature_list_appendable]
            if set == "ticker":
                feature_list.append("Momentum_14_ticker_finiteautomata_sentiment")
            else:
                feature_list.append("Momentum_14_product_finiteautomata_sentiment")
            feature_list.extend(["Real Volume","MOM_14","Volatility","RSI_14"])

            predict_return = afunc(feature_list, coin, set, data_df, predict_return)

            #predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)
            #predict_return.to_csv(r'return_logreg_predictions.csv', index = False)

            #predict_return = SVM_Pred(feature_list, coin, set, data_df, predict_return)
            #predict_return.to_csv(r'return_SVM_predictions.csv', index = False)

            #predict_return = KNN_Pred(feature_list, coin, set, data_df, predict_return)
            #predict_return.to_csv(r'return_KNN_predictions.csv', index = False)


        print("-------------------------------------")

        """
        #---------------------------------
        #run with sentiment features only
        feature_list_appendable = ["_number_of_tweets", "_average_number_of_likes", "_average_number_of_retweets", "_average_number_of_followers", "_finiteautomata_sentiment","_finiteautomata_sentiment_expectation_value_volatility"]
        feature_list = [set + item for item in feature_list_appendable]
        if set == "ticker":
            feature_list.extend(("ROC_2_ticker_finiteautomata_sentiment","Momentum_14_ticker_finiteautomata_sentiment"))
        else:
            feature_list.extend(("ROC_2_product_finiteautomata_sentiment","Momentum_14_product_finiteautomata_sentiment"))

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)

        #---------------------------------
        #run with finance features only
        feature_list = ["Real Volume","Circulating Marketcap", "Sharpe Ratio", "Volatility", "MOM_14","RSI_14","pos_conf","neg_conf"]

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)

        #---------------------------------
        #run with network features only
        if coin == "BTC":
            feature_list = ["Adjusted NVT","Adjusted RVT", "Deposits on Exchanges", "Withdrawals from Exchanges", "Average Transaction Fees", "Adjusted Transaction Volume", "Average Transfer Value", "Active Supply", "Miner Supply", "Miner Revenue per Hash per Second", "Addresses Count", "Active Addresses Count", "Addresses with balance greater than $1"]

        elif coin == "ADA":
            feature_list = ["Adjusted NVT","Adjusted RVT", "Average Transaction Fees", "Adjusted Transaction Volume", "Average Transfer Value", "Active Supply", "Addresses Count", "Active Addresses Count", "Addresses with balance greater than $1"]

        predict_return = LogReg_Pred(feature_list, coin, set, data_df, predict_return)

        #---------------------------------
        """
