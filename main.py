'''
File: main.py
Description: Generate a lsit of songs for a user to try based off Spotify playlists
Author: Devin Lepur
Date: 06/28/2024
'''


import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from spotify_client import get_spotify_access_token, get_playlist_track_ids
from lyrics import get_lyrics
from data_cleaning import get_track_data, clean_track_data
from sentiment import append_sentiment
from model_generation import get_user_model


# Load enviornment variables
load_dotenv()

# Get credentials from enviornment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def main():
    
    # Obtain an access token
    token = get_spotify_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    target_playlist_id = '37i9dQZF1DWTl4y3vgJOXW'       # Example target playlist "Locked In" by Spotify
    unknown_playlist_id = '7x0UDYDd2MWVkBEGGyPZbm'      # Example unkown playlsit "Biggest Songs of All Time Top500" by olaf_aarts

    # Get track IDs
    target_track_ids = get_playlist_track_ids(target_playlist_id, token)
    unknown_track_ids = get_playlist_track_ids(unknown_playlist_id, token)

    # Get audio features and artist, song title, popularity
    target_track_data = get_track_data(target_track_ids, token)
    unknown_track_data = get_track_data(unknown_track_ids, token)

    # Create binary istarget column
    target_track_data['is_target'] = 1
    unknown_track_data['is_target'] = 0

    # Combine and clean data
    merged_df = pd.concat([target_track_data, unknown_track_data])
    merged_df = clean_track_data(merged_df)

    # Get lyric sentiment
    merged_df = append_sentiment(merged_df)



    # Train model
    train, test = train_test_split(merged_df, test_size=0.20, random_state=42)
    model, features = get_user_model(train)
    print("Features used:", np.array(features))

    # Filter test set by features used in the model
    filtered_test = test.loc[:, features]
    X_test = filtered_test.drop(columns=['is_target'])

    # Predict song classification
    y_pred = model.predict(X_test)

    # Create a new column for predictions on the test set
    test = test.copy()
    test['pred_label'] = y_pred

    # Print test cases where is_target is 0 and pred_label is 1
    for index, song in test.iterrows():
        if (song['is_target'] == 0) and (song['pred_label'] == 1):
            print(f"Try this song: {song['title']}, by: {song['main_artist']}")
    


if __name__ == "__main__":
    main()