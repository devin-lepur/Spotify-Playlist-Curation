import os
import requests
import pandas as pd
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
    endpoint = f'https://api.spotify.com/v1/audio-features'
    params = {
        'ids': ','.join(track_ids)
    }

    return get_spotify_data(endpoint, access_token)



def main():
    # Obtain an access token
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)

    # Example playlist ID
    playlist_id = '0zVYBTlgm6tBItMJ9smASQ'

    # Fetch tracks from the playlist
    playlist_data = get_playlist_tracks(playlist_id, token)
    tracks = playlist_data['items']

    # Extract track IDs
    track_ids = [track['track']['id'] for track in tracks if track['track']]

    # Fetch audio features for the tracks
    audio_features_data = get_audio_features(track_ids, token)
    
    #DEBUGGING
    print(audio_features_data)

    audio_features = audio_features_data['audio_features']

    # Create a DataFrame from the audio features
    df = pd.DataFrame(audio_features)

    # Save the DataFrame to a CSV file
    df.to_csv('spotify_audio_features.csv', index=False)

    print("Data exported to spotify_audio_features.csv")

if __name__ == "__main__":
    main()