'''
File: data_cleaning.py
Description: Combine and clean data sources
Author: Devin Lepur
Date: 06/30/2024
'''

import pandas as pd

from spotify_client import  get_audio_features, get_title_artist


def get_track_data(track_ids, access_token):
    """
    Get all data to be used in a dataframe from track ids

    track_ids (list[str]): list of track IDs
    access_token (str): Access token for Spotify authorization

    Returns:
    pd.DataFrame: df containing audio features plus track title and artist
    """
    # Output to display where program is at
    print("Getting tracks' Spotify data...")

    audio_features = get_audio_features(track_ids, access_token)
    title_artist = get_title_artist(track_ids, access_token)

    combined_df = pd.merge(audio_features, title_artist, on='id')

    print("Tracks' Spotify data collected.")
    return combined_df



def clean_track_data(df):
    """
    Clean a provided dataframe of missing values and repeats
    
    df (pd.DataFrame): dataframe of audio features and song info
    
    Returns:
    pd.DataFrame: cleaned df ready for exploration or models
    """
    # Output to display where program is at
    print("Cleaning track data...")

    #Drop duplicate songs
    df.drop_duplicates(subset=['id'], inplace=True)
    df.dropna(inplace=True)


    # Drop unnecessary audio features
    df.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'], inplace=True)

    # Convert duration_ms to minutes
    df['duration_ms'] = df['duration_ms'] / 60000
    df.rename(columns={'duration_ms':'duration_min'}, inplace=True)

    print("Track data cleaned")
    return df