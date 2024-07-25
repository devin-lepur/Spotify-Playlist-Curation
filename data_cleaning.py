'''
File: data_cleaning.py
Description: Combine and clean data sources
Author: Devin Lepur
Date: 06/30/2024
'''

import pandas as pd
from datetime import datetime
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

    #Drop duplicate songs based off id and then also title and artist together
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['id'], inplace=True)
    df.drop_duplicates(subset=['title', 'main_artist'], inplace=True)
    df.dropna(inplace=True)


    # Drop unnecessary audio features
    df.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url'], inplace=True)

    # Convert duration_ms to minutes
    df['duration_ms'] = df['duration_ms'] / 60000
    df.rename(columns={'duration_ms':'duration_min'}, inplace=True)

    # Convert release_date to years since release
    DAYS_IN_YEAR = 365.25
    today = datetime.now()

    df['release_date'] = pd.to_datetime(df['release_date'], format='mixed')
    df['years_since_release'] = (today - df['release_date']).dt.days / DAYS_IN_YEAR
    df.drop(columns=['release_date'], inplace=True)

    print("Track data cleaned")
    return df


def clean_features(df):
    """
    Performes feature removal, scaling, etc. on data and returns new dataframe
    
    df(pd.DataFrame): dataframe to prepare for training
    
    Returns:
    pd.DataFrame: DataFrame with adjustments made
    """

    # Remove rows with missing values (Likely redundant)
    df.dropna(inplace=True)

    # Remove features deemed unfit for training
    columns_to_remove = ['mode', 'time_signature', 'loudness', 'neu', 'compound']
    df.drop(columns=columns_to_remove, inplace=True)

    # Spread data distributions for features with skew near 0
    df['instrumentalness'] = df['instrumentalness']**(1/3)
    df['acousticness'] = df['acousticness']**(1/3)
    df['liveness'] = df['liveness']**(1/3)
    df['pos'] = df['pos']**(1/3)
    df['neg'] = df['neg']**(1/3)
    df['speechiness'] = df['speechiness']**(1/3)

    # Spread data distribitions for features with skew near |1|
    df['danceability'] = df['danceability']**3
    df['energy'] = df['energy']**3



    return df