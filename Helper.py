import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def RFE_logistic_regression(X, y, regression='l1', variance_pct=0.0,
                            verbose=False, random_state=0) -> (list, float):
    """
    This function implements the recursive forward elimination method for feature selection. In this implementation,
    logistic regression is applied to the complete feature set and then the least important feature is dropped.
    This process continues until we get the optimal number of features.

    :param X: Feature set
    :param y: target
    :param regression: Penalty parameter for logistic regression ['l1', 'l2']
    :param variance_pct: The allowable % reduction in the accuracy after each iteration
    :param verbose: True or False
    :param random_state: Default = 0
    :return: reduced feature set
    """
    precision = 0
    recall = 0
    f1 = 0
    cm = np.array([[0, 0], [0, 0]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    previous_accuracy = -1
    new_accuracy = 0
    if (regression.lower() != 'l1') and (regression.lower() != 'l2'):
        print('Regression must be either l1 or l2.\n Model defaulted to l1')
        regression = 'l1'
    while previous_accuracy * (1 - variance_pct) < new_accuracy:
        previous_accuracy = new_accuracy
        lasso_model = LogisticRegression(penalty=regression, solver='liblinear', random_state=random_state)
        lasso_model.fit(X_train, y_train)
        lasso_pred = lasso_model.predict(X_test)
        cm = confusion_matrix(y_test, lasso_pred)
        new_accuracy = accuracy_score(y_test, lasso_pred)
        precision = precision_score(y_test, lasso_pred)
        recall = recall_score(y_test, lasso_pred)
        f1 = f1_score(y_test, lasso_pred)
        X_train_cols = X_train.columns
        drop_feature = abs(pd.DataFrame(lasso_model.coef_, columns=X_train_cols)).idxmin(axis=1)[0]
        X_train.drop(columns=[drop_feature], inplace=True)
        X_test.drop(columns=[drop_feature], inplace=True)
    keep_cols = X_train.columns.tolist()

    if verbose:
        print(keep_cols)
        print(f"\n\naccuracy after feature drop: {new_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.yticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.show()

    return keep_cols, new_accuracy


def logistic_regression(X, y, regression='l1', verbose=False) -> (float, float, float, float):
    """
    This function implements the logistic regression ['l1', 'l2']. It returns the accuracy, precision, recall and f1.
    :param X: Feature set
    :param y: The target
    :param regression: logistic regression ['l1', 'l2']
    :param verbose: True or False
    :return: Accuracy, precision, recall, f1
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    if (regression.lower() != 'l1') and (regression.lower() != 'l2'):
        print('Regression must be either l1 or l2.\n Model defaulted to l1')
        regression = 'l1'

    lasso_model = LogisticRegression(penalty=regression, solver='liblinear')
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    cm = confusion_matrix(y_test, lasso_pred)
    accuracy = accuracy_score(y_test, lasso_pred)
    precision = precision_score(y_test, lasso_pred)
    recall = recall_score(y_test, lasso_pred)
    f1 = f1_score(y_test, lasso_pred)

    if verbose:
        print('\n\naccuracy after feature drop: ', accuracy)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.yticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.show()

    return accuracy, precision, recall, f1


def random_forest(X, y, n_estimators, max_depth, verbose=False) -> (float, float, float, float):
    """
    This function implements random forest classification and returns the accuracy, precision, recall and f1.
    :param X: Feature set
    :param y: The target
    :param n_estimators: Number of trees
    :param max_depth: maximum depth of trees
    :param verbose: True or False
    :return: Accuracy, precision, recall, f1
    """
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.3)
    clf_rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf_rf.fit(X_train_rf, y_train_rf)
    y_pred_rf = clf_rf.predict(X_test_rf)
    accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
    cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
    precision_rf = precision_score(y_test_rf, y_pred_rf)
    recall_rf = recall_score(y_test_rf, y_pred_rf)
    f1_rf = f1_score(y_test_rf, y_pred_rf)

    if verbose:
        print(f"Accuracy: {accuracy_rf:.4f}")
        print(f"Precision: {precision_rf:.4f}")
        print(f"Recall: {recall_rf:.4f}")
        print(f"F1 Score: {f1_rf:.4f}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.yticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.show()

    return accuracy_rf, precision_rf, recall_rf, f1_rf


def xgboost_classifier(X, y, n_estimators, max_depth, verbose=False) -> (float, float, float, float):
    """
    This function implements XGBoost classification and returns the accuracy, precision, recall and f1.
    :param X: Feature set
    :param y: The target
    :param n_estimators: Number of trees
    :param max_depth: maximum depth of trees
    :param verbose: True or False
    :return: Accuracy, precision, recall, f1
    """
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.3)
    clf_xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
    clf_xgb.fit(X_train_xgb, y_train_xgb)
    y_pred_xgb = clf_xgb.predict(X_test_xgb)
    accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
    cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
    precision_xgb = precision_score(y_test_xgb, y_pred_xgb)
    recall_xgb = recall_score(y_test_xgb, y_pred_xgb)
    f1_xgb = f1_score(y_test_xgb, y_pred_xgb)

    if verbose:
        print(f"Accuracy: {accuracy_xgb:.4f}")
        print(f"Precision: {precision_xgb:.4f}")
        print(f"Recall: {recall_xgb:.4f}")
        print(f"F1 Score: {f1_xgb:.4f}")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.yticks(ticks=[0.5, 1.5], labels=['Graduate', 'Dropout'])
        plt.show()

    return accuracy_xgb, precision_xgb, recall_xgb, f1_xgb
