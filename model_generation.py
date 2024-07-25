'''
File: model_generation.py
Description: Create a machine learning model for the user
Author: Devin Lepur
Date: 07/12/2024
'''

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from data_cleaning import clean_features
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def get_important_features(model, columns):
    """
    Return features with importance above a threshold
    
    model: An XGBoost model capable of feature_importances_
    columns (list[str]): columns the model was trained on
    
    Returns:
    list[str] list of column names with unimportant features removed
    """
    FEATURE_THRESHOLD = 0.1
    
    feature_importance = list(zip(columns, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1])

    # If worst feature is below threshold, return all other features
    if (feature_importance[0][1] < FEATURE_THRESHOLD):
        important_features = [feature[0] for feature in feature_importance[1:]]
        return important_features
    else:
        # Features deemed important enough, return all    
        return columns

def create_model(df):
    """
    Create a model off the data provided
    
    df (pd.DataFrame): Dataframe of song data with all features already cleaned
    
    Returns:
    XGBoost model trained off the data 
    """

    # Threshold for reliable negatives
    NEG_THRESHOLD = 0.2

    X = df.drop(columns=['is_target'])
    y = df['is_target']

    # Init classifier
    clf = XGBClassifier(eta=0.8, subsample=0.5, reg_alpha=0.1, reg_lambda=0.1)

    # Fit classifier, assume 0s initially are negatives
    clf.fit(X, y)

    # Get probability of sample being positive
    y_pred_proba = clf.predict_proba(X)

    # Get reliable negatives
    reliable_negatives = X[y_pred_proba[:, 0] < NEG_THRESHOLD]
    reliable_negative_labels = np.zeros(reliable_negatives.shape[0])

    # Get data where is_target == 1
    positive_samples = df.loc[X.index][df['is_target'] == 1].drop(columns=['is_target'])
    positive_labels = np.ones(positive_samples.shape[0])

    # Merge positives and negatives
    X_merged = pd.concat([positive_samples, reliable_negatives])
    y_merged = np.concatenate([positive_labels, reliable_negative_labels])

    # Oversample minority class using SMOTE
    smote = SMOTE(sampling_strategy='minority')
    X_over, y_over = smote.fit_resample(X_merged, y_merged)

    # Retrain classifier
    clf.fit(X_over, y_over)

    return clf

def get_user_model(df):
    """
    Get the best model for the user's data
    
    Returns:
    XGBoost model: trained off the user's data
    columns (list[str]): List of the columns kept for use predicting
    """

    # Clean, normalize, scale data, and remove useless labels
    df = clean_features(df)

    df.drop(columns=['title', 'main_artist'], inplace=True)

    model = create_model(df)

    # Rename for loop purposes
    filtered_df = df
    
    # Loop until all features are deemed important
    while True:
        current_features = filtered_df.drop(columns=['is_target']).columns
        new_features = get_important_features(model, current_features)
        if len(current_features) == len(new_features):
            break

        # Ensure is_target column is not lost
        filtered_df = filtered_df.loc[:, new_features + ['is_target']]
        model = create_model(filtered_df)

    return model, filtered_df.columns

