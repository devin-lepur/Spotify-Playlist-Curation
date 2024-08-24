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



def create_model(df):
    """
    Create a model off the data provided
    
    df (pd.DataFrame): Dataframe of song data with all features already cleaned
    
    Returns:
    XGBoost model trained off the data 
    """

    # Threshold for reliable negatives
    NEG_THRESHOLD = 0.1

    # Separate features and target
    X = df.drop(columns=['is_target'])
    y = df['is_target']

    # Init classifier
    clf = XGBClassifier(eta=0.1, subsample=0.5, reg_alpha=0.1, reg_lambda=0.1)

    # Fit classifier, assume 0s initially are negatives
    clf.fit(X, y)

    # Get probability of sample being positive
    y_pred_proba = clf.predict_proba(X)

    # Get reliable negatives with a probability of being positive <= NEG_THRESHOLD
    reliable_negatives = X[y_pred_proba[:, 1] <= NEG_THRESHOLD]
    reliable_negative_labels = np.zeros(reliable_negatives.shape[0])

    # Get data where is_target == 1
    positive_samples = df[df['is_target'] == 1].drop(columns=['is_target'])
    positive_labels = np.ones(positive_samples.shape[0])

    # Merge positives and negatives
    X_merged = pd.concat([positive_samples, reliable_negatives], ignore_index=True)
    y_merged = np.concatenate([positive_labels, reliable_negative_labels])

    # Oversample minority class using SMOTE
    #smote = SMOTE(sampling_strategy='minority')
    #X_over, y_over = smote.fit_resample(X_merged, y_merged)

    # Retrain classifier
    clf.fit(X_merged, y_merged)

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

    return model, filtered_df.columns