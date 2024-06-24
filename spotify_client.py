import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load enviornemnt variables
load_dotenv()

# Spotify credentials from .env file
CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Spotify Accounts service API URL
TOKEN_URL = 'https://accounts.spotify.com/api/token'

def get_access_token(client_id, client_secret):
    """
    Get Access token using client credentials flow
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
    Get tracks from a Spotify playlist id.
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
    Get audio features for a list of track IDs.
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

    


def clean_audio_features(likedDF, dislikedDF):
    """
    Clean the dataframes provided and return merged dataframe
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

    return mergedDF

def logistic_reg():
    """
    Train a model to predict whether a song would be liked or not
    """



def main():
    # Obtain an access token
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    liked_playlist_id = '6kBzzBza7wIPtOykytjABq'
    disliked_playlist_id = '3caseqKMvJyv2XE1rN6SQi'

    # Get track IDs
    liked_track_ids = get_playlist_track_ids(liked_playlist_id, token)
    disliked_track_ids = get_playlist_track_ids(disliked_playlist_id, token)

    # Get track audio features
    liked_audio_features = get_audio_features(liked_track_ids, token)
    disliked_audio_features = get_audio_features(disliked_playlist_id, token)

    #Convert to dataframe
    likedDF = pd.DataFrame(liked_audio_features)
    dilikedDF = pd.DataFrame(disliked_audio_features)
    print(disliked_audio_features)

    #TODO clean and merge dataframes



if __name__ == "__main__":
    main()