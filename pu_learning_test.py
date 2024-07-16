'''
File: pu_learning_test.py
Description: Test the ability of positive unlabled learning on the dataset
Author: Devin Lepur
Date: 07/12/2024
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


df = pd.read_csv("tester.csv")

df.drop(columns=['pos', 'neg', 'neu', 'time_signature', 'mode', 'key', 'instrumentalness', 'loudness'], inplace=True)

X = df.drop(columns=['isLiked', 'main_artist', 'title'])
y = df['isLiked']

# Split data into training and test, random_state set so test data aligns with manually made labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace y_test with manually labeled data
labeled_test_data = pd.read_csv("labeled_test.csv") 
y_test = labeled_test_data['truth_value']

# Init classifier
clf = RandomForestClassifier()

# Fit classifier
clf.fit(X_train, y_train)

# Predict probabilities of sample being positive
Y_pred_proba = clf.predict_proba(X_test)[:,1]

# Threshold for reliable negatives
THRESHOLD = 0.1

# Get reliable negatives and combine with positive samples
reliable_negatives = X_test[Y_pred_proba < THRESHOLD]
reliable_negative_labels = np.zeros(reliable_negatives.shape[0])

# Get X_train data where isLiked == 1
positive_samples = df.loc[X_train.index][df['isLiked'] == 1].drop(columns=['isLiked', 'main_artist', 'title'])
positive_labels = np.ones(positive_samples.shape[0])

X_merged = pd.concat([positive_samples, reliable_negatives])
y_merged = np.concatenate([positive_labels, reliable_negative_labels])

print(len(X_merged))
print(len(y_merged))


# Retrain classifier
clf.fit(X_merged, y_merged)

y_pred = clf.predict(X_test)

# Output metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(confusion_matrix(y_test, y_pred))