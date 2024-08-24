'''
File: lyrics.py
Description: Communicate with Genius API to extract and clean song lyrics
Author: Devin Lepur
Date: 06/28/2024
'''


import os
import requests
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time



# Load enviornment labels
load_dotenv()

#Get credential from enviornment variables
GENIUS_API_TOKEN = os.getenv('GENIUS_API_TOKEN')


def get_genius_access_token():
    '''
    Get Genius API token from enviornment variables

    Returns:
    str: Genius API token
    '''

    token = GENIUS_API_TOKEN
    if not token:
        raise Exception("Genius API token not found in enviorment variables")
    return token



def remove_song_labels(text):
    '''
    Remove all text within square brackets, including the brackets

    text (str): Song lyrics from Genius with chorus and verse labels

    Returns:
    str: Song lyrics without any chorus or verse labels
    '''

    # Re expression to remove square brackets
    pattern = r'\[.*?\]'

    # Replace all occurances of pattern with empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Remove any extra spaces resulting from the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text



def get_lyrics(song_title, artist_name):
    """
    Get song lyrics from Genius API

    song_title (str): title of song, can be case sensative
    artist_name (str): name of artist, can be case sensative

    Returns:
    str: Clean song lyrics with no newlines or labels
    """
    
    base_url = 'https://api.genius.com'
    search_url = f'{base_url}/search'
    headers = {'Authorization': f'Bearer {get_genius_access_token()}'}
    
    # Search for the song
    search_params = {'q': f'{song_title} {artist_name}'}
    response = requests.get(search_url, params=search_params, headers=headers)


    # Check the status code
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        retry_after = int(response.headers.get("Retry-After", 60))
        print(f"Rate limit hit. Retry after {retry_after} seconds.")
        return "No lyrics Found"
    
    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: JSONDecodeError - Response content is not valid JSON")
        print("Response content:", response.text)
        return None
    
    # Check if there were results
    if response_data['response']['hits']:
        song_path = response_data['response']['hits'][0]['result']['path']
        song_url = f'https://genius.com{song_path}'
        lyrics_page = requests.get(song_url)

        # Extract lyrics from the song's page
        soup = BeautifulSoup(lyrics_page.text, 'html.parser')
        
        # Find all lyrics containers
        lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
        lyrics_texts = []
        
        for container in lyrics_containers:
            if container:
                lyrics_texts.append(container.get_text(separator='\n').strip())
        
        # Join all collected text
        if lyrics_texts:
            return remove_song_labels("\n".join(lyrics_texts).strip())
        
        return "Lyrics not found"
    else:
        return "Song not found"
