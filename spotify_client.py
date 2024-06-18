import os
import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
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

    if response.status_code != 200:
        raise Exception("Could not fetch data from Spotify API")
    
    return response.json()



def get_playlist_tracks(playlist_id, access_token):
    """
    Get tracks from a Spotify playlist.
    """
    endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    return get_spotify_data(endpoint, access_token)



def get_audio_features(track_ids, access_token):
    """
    Get audio features for a list of track IDs.
    """
    track_ids = ','.join(track_ids)
    endpoint = f'https://api.spotify.com/v1/audio-features?ids={track_ids}'
    
    return get_spotify_data(endpoint, access_token)


def clean_audio_features(likedDF, dislikedDF):
    """
    Clean the data provided
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

    # TODO: Perform log scaling on duration_ms and min_max on tempo

    mergedDF.info()

    return mergedDF

def logistic_reg():
    """
    Train a model to predict whether a song would be liked or not
    """



def main():
    # Obtain an access token
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    liked_playlist_id = '1Y5qoloOrSwRoNEkTCsglp'
    disliked_playlist_id = '3IVyfenjdTnuqP9OLUqPcR'

    # Fetch tracks from the playlist
    liked_playlist_data = get_playlist_tracks(liked_playlist_id, token)
    liked_tracks = liked_playlist_data['items']

    disliked_playlist_data = get_playlist_tracks(disliked_playlist_id, token)
    disliked_tracks = disliked_playlist_data['items']

    # Extract track IDs
    liked_track_ids = [track['track']['id'] for track in liked_tracks if track['track']]
    disliked_track_ids = [track['track']['id'] for track in disliked_tracks if track['track']]

    # Fetch audio features for the tracks
    liked_features = get_audio_features(liked_track_ids, token)
    disliked_features = get_audio_features(disliked_track_ids, token)
    
    liked_features = liked_features['audio_features']
    disliked_features = disliked_features['audio_features']


    # Create a DataFrame from the audio features
    likedDF = pd.DataFrame(liked_features)
    dislikedDF = pd.DataFrame(disliked_features)

    df = clean_audio_features(likedDF, dislikedDF)
    print(df)

    print("Data exported to spotify_audio_features.csv")

if __name__ == "__main__":
    main()