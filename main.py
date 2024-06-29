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

from spotify_client import(
    get_spotify_access_token, get_spotify_data, get_playlist_track_ids, 
    get_audio_features, clean_merge_features
    )
from lyrics import get_lyrics


# Load enviornment variables
load_dotenv()

# Get credentials from enviornment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def main():
    #Example lyric obtaining
    song = 'good kid'
    artist = 'Kendrick Lamar'
    print(get_lyrics(song, artist))

    # Obtain an access token
    token = get_spotify_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    liked_playlist_id = '6kBzzBza7wIPtOykytjABq'
    disliked_playlist_id = '3caseqKMvJyv2XE1rN6SQi'

    # Get track IDs
    liked_track_ids = get_playlist_track_ids(liked_playlist_id, token)
    disliked_track_ids = get_playlist_track_ids(disliked_playlist_id, token)

    # Get track audio features
    liked_audio_features = get_audio_features(liked_track_ids, token)
    disliked_audio_features = get_audio_features(disliked_track_ids, token)

    #Convert to dataframe
    liked_df = pd.DataFrame(liked_audio_features)
    disliked_df = pd.DataFrame(disliked_audio_features)
    
    # Clean and merge dataframes and create a isLiked column
    collective_data = clean_merge_features(liked_df, disliked_df)


    #VIF for colinear features
    '''
    X = collective_data.drop('isLiked', axis=1)
    vif=pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Run logistic regression
    logistic_reg(collective_data)
    
    print(vif)
    '''



if __name__ == "__main__":
    main()