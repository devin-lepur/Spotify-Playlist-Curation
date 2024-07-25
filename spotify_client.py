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
import tqdm
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
    params = {
        'market': 'US'
    }
    response = requests.get(endpoint, headers=headers, params=params)

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

    # Output to display where program is at
    print("Getting playlist track IDs...")

    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'

    results = get_spotify_data(endpoint, access_token)

    # Iterate through data for each 100 songs until there is no more next song
    tracks = results['items']
    while results['next']:
        results = get_spotify_data(results['next'], access_token)
        tracks.extend(results['items'])
    # Extract ids
    track_ids = [track['track']['id'] for track in tracks if track['track'] and track['track']['id']]

    # Confirm completion
    print("Playlist track IDs fetched.")
    return track_ids



def get_audio_features(track_ids, access_token):
    """
    Get audio features for a list of track IDs

    track_ids (list[str]): List of Spotify track ids
    access_token (str): Access token for Spotify authorization

    Returns:
    pd.DataFrame: Dataframe of track's audio features as each row
    """

    # Spotify limits requests to a size of 100
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

    return pd.DataFrame(filtered_audio_features)

    

def get_title_artist(track_ids, access_token):
    """
    Get song title and artist for a given track ID
    
    track_ids (list[str]): list of track IDs
    access_token (str): Access token for Spotify authorization
    
    Returns:
    pd.DataFrame: A dataframe of track id, title, and artist
    """

    # Spotify limits requests to a size of 50
    MAX_BATCH = 50

    # List to store track info
    track_info = []

    # Iterate through batches of 50 and make seperate requests
    for i in range(0, len(track_ids), MAX_BATCH):
        batch = track_ids[i:i+MAX_BATCH]
        ids = ','.join(batch)
        endpoint = f'https://api.spotify.com/v1/tracks?ids={ids}'
        response = get_spotify_data(endpoint, access_token)
        tracks = response.get('tracks', [])

        if tracks is None:
            # Handle the case where 'tracks' is None
            for track_id in batch:
                track_info.append({'id': track_id, 'title': None, 'main_artist': None, 
                                       'popularity': None, 'release_date': None})
        else:
            for track in tracks:
                if track is not None: 
                    # Fetch features for each track
                    title = track.get('name')
                    main_artist = track['artists'][0]['name'] if track['artists'] else None
                    track_id = track.get('id')
                    popularity = track.get('popularity')
                    release_date = track.get('album',{}).get('release_date')

                    # Create dict for each track and store in list
                    track_info.append({
                        'id': track_id, 
                        'title': title, 
                        'main_artist': main_artist, 
                        'popularity': popularity,
                        'release_date': release_date
                        })
                else:
                    # Handle the case where track is None
                    track_info.append({'id': track_id, 'title': None, 'main_artist': None, 
                                       'popularity': None, 'release_date': None})

    # Convert list of dicts to DataFrame with column names
    df = pd.DataFrame(track_info, columns=['id', 'title', 'main_artist', 'popularity', 'release_date'])
    return df



