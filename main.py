'''
File: main.py
Description: Communicate with Genius API to extract and clean song lyrics
Author: Devin Lepur
Date: 06/28/2024
'''


import os
import requests
import pandas as pd
from dotenv import load_dotenv
from statsmodels.stats.outliers_influence import variance_inflation_factor
import cProfile
import pstats

from spotify_client import get_spotify_access_token, get_playlist_track_ids
from lyrics import get_lyrics
from data_cleaning import get_track_data, clean_track_data
from sentiment import append_sentiment


# Load enviornment variables
load_dotenv()

# Get credentials from enviornment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def main():
    # TODO: 
    #   - Fix insane load time for song_sentiment maybe threading
    #   - Ensure integretity even when lyrics aren't found
    #   - Get Speechiness from Spotify
    #   - Get valance from Spotify
    #   - Progress bars/reports for certain processes to update user
    #   - TODO (Later): Explore Positive Unlabeled Machine Learning

    # Commented out for testing purposes

    
    # Obtain an access token
    token = get_spotify_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    liked_playlist_id = '6kBzzBza7wIPtOykytjABq'
    disliked_playlist_id = '3caseqKMvJyv2XE1rN6SQi'

    # Get track IDs
    liked_track_ids = get_playlist_track_ids(liked_playlist_id, token)
    disliked_track_ids = get_playlist_track_ids(disliked_playlist_id, token)

    # Get audio features with artist and song title
    liked_track_data = get_track_data(liked_track_ids, token)
    disliked_track_data = get_track_data(disliked_track_ids, token)

    # Create binary isLiked column
    liked_track_data['isLiked'] = 1
    disliked_track_data['isLiked'] = 0

    merged_df = pd.concat([liked_track_data, disliked_track_data])

    merged_df = clean_track_data(merged_df)

    merged_df

    merged_df = append_sentiment(merged_df)

    merged_df.to_csv("tester.csv")

    # TODO create an evaluation data set comprised of about 20% of the train data size 
    #   that will be seperated from training data and determine truth values for these

if __name__ == "__main__":
    main()