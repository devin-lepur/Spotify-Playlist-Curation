'''
File: sentiment.py
Description: Provide text cleaning and sentiment analysis
Author: Devin Lepur
Date: 06/29/2024
'''

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from lyrics import get_lyrics



class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, lyrics):
        sentiment = self.analyzer.polarity_scores(lyrics)
        return sentiment



def append_sentiment(df):
    """
    Add sentiment columns to dataframe

    df (pd.DataFrame): cleaned track dataframe

    Returns:
    pd.DataFrame: parameter dataframe with sentiment analysis columns added
    """
    # Output to display where program is at
    print("Appending sentiment...")


    # Create columns to fill information with default value None
    df['neg'] = None
    df['neu'] = None
    df['pos'] = None
    df['compound'] = None

    sentiment_analyzer = SentimentAnalyzer()

    # function to process each row
    def process_row(idx, track):
        lyrics = get_lyrics(track['title'], track['main_artist'])
        sentiment_scores = sentiment_analyzer.analyze(lyrics)
        return idx, sentiment_scores

    # Use threads to parallelize the processing with default workers
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_row, idx, track): idx for idx, track in df.iterrows()}
        
        # Display progress bar
        for future in tqdm(as_completed(futures), desc='Append Sentiment Progress', total=df.shape[0]):
            idx, sentiment_scores = future.result()
            df.at[idx, 'neg'] = sentiment_scores['neg']
            df.at[idx, 'neu'] = sentiment_scores['neu']
            df.at[idx, 'pos'] = sentiment_scores['pos']
            df.at[idx, 'compound'] = sentiment_scores['compound']
            continue

    df['neg'] = df['neg'].astype(float)
    df['pos'] = df['pos'].astype(float)
    print("Sentiment appended.")
    return df