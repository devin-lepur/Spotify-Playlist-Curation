'''
File: spotify_client.py
Description: Communicate with Spotify's API
Author: Devin Lepur
Date: 05/20/2024
'''

import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from dotenv import load_dotenv

from lyrics import get_lyrics



# Load enviornemnt variables
load_dotenv()

# Get credentials from enviornment variables
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Spotify Accounts service API URL
TOKEN_URL = 'https://accounts.spotify.com/api/token'


def get_spotify_access_token(client_id, client_secret):
    """ 
    Get Access token using client credentials flow

    client_id (str): Client ID as provided by Spotify for application
    client_secret (str): Client Secret as provided by Spotify for application

    Returns:
    str: Token for authorization
    """

    auth_response = requests.post(TOKEN_URL, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    })

    if auth_response.status_code != 200:
        raise Exception("Could not authenticate with Spotify API")
    
    auth_response_data = auth_response.json()
    return auth_response_data['access_token']



def get_spotify_data(endpoint, access_token):
    """
    Make a request to the Spotify API

    endpoint (str): URL containing the endpoint for data
    access_token (str): Access token for Spotify authorization

    Returns
    JSON: Data fetched from the endpoint
    """

    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(endpoint, headers=headers)

    # If unsucessful print status code, content, and raise exception
    if response.status_code != 200:
        print(f"Request to {endpoint} failed with status code {response.status_code}")
        print(f"Response content: {response.text}")
        raise Exception("Could not fetch data from Spotify API")
    
    return response.json()



def get_playlist_track_ids(playlist_id, access_token):
    """
    Get tracks from a Spotify playlist ID

    playlist_id (str): ID for a given playlist from URL or Spotify request
    access_token (str): Access token for Spotify authorization

    Returns:
    list[str]: List of Spotify track IDs
    """

    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'

    results = get_spotify_data(endpoint, access_token)

    # Iterate through data for each 100 songs until there is no more next song
    tracks = results['items']
    while results['next']:
        results = get_spotify_data(results['next'], access_token)
        tracks.extend(results['items'])
    # Extract ids
    track_ids = [track['track']['id'] for track in tracks if track['track'] and track['track']['id']]
    return track_ids



def get_audio_features(track_ids, access_token):
    """
    Get audio features for a list of track IDs

    track_ids (list[str]): List of Spotify track ids
    access_token (str): Access token for Spotify authorization

    Returns:
    list[dict{str|int}]: One list with dicts for every songs' features
    """

    # Spotify limits request to a size of 100
    MAX_BATCH = 100

    # Iterate through batches of 100 and make seperate requests
    audio_features = []
    for i in range(0, len(track_ids), MAX_BATCH):
        batch = track_ids[i:i+MAX_BATCH]
        ids = ','.join(batch)
        endpoint = f'https://api.spotify.com/v1/audio-features?ids={ids}'
        response = get_spotify_data(endpoint, access_token)
        audio_features.extend(response.get('audio_features', []))

    
    # Remove any songs without audio features available
    filtered_audio_features = []
    for obj in audio_features:
        if obj != None:
            filtered_audio_features.append(obj)
    

    return filtered_audio_features

    


def clean_merge_features(likedDF, dislikedDF):
    """
    Clean data frames and merge features of liked and disliked dataframes

    likedDF (pd.DataFrame): DataFrame of liked song features to be labeled liked
    dislikedDF (pd.DataFrame): DataFrame of disliked song features to be labeled disliked

    Returns:
    pd.DataFrame: DataFrame of merged data with isLiked column
    """

    #Create boolean feature; 0: liked, 1: not liked
    likedDF['isLiked'] = 1
    dislikedDF['isLiked'] = 0

    #Merge columns and drop second instance of any duplicate, i.e. any duplicates from disliked
    mergedDF = pd.concat([likedDF, dislikedDF])
    mergedDF.drop_duplicates(inplace=True)

    # Drop useless, non numeric columns and any resulting rows with null values
    mergedDF.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'], inplace=True)
    mergedDF.dropna(inplace=True)

    # Convert duration_ms to minutes
    mergedDF['duration_ms'] = mergedDF['duration_ms'] / 1000 / 60
    mergedDF.rename(columns={'duration_ms': 'minutes'}, inplace=True)

    # Drop features with VIF > 5
    mergedDF.drop(columns=['danceability', 'energy', 'loudness', 'valence', 'tempo', 'minutes', 'time_signature'], inplace=True)

    return mergedDF


# Probably will get rid of this in a later version
def logistic_reg(df):
    """
    Train a model to predict whether a song would be liked or not
    """
    X = df.drop('isLiked', axis=1)
    y = df['isLiked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Logistic Regression f_1 score: {f1_score(y_test, y_pred)}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}")





