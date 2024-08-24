Spotify Playlist Curation Using Positive-Unlabeled Machine Learning
========

Devin Lepur

8/24/2024

Abstract
========

In this project, I initially aimed to recommend songs to a user based on their liked songs and a larger playlist to recommend from using the Spotify API. However, due to limitations in the data available, the focus evolved towards developing a machine learning model that tailors personalized playlists. This shift allows for a more expansive use of the model, which can benefit from common themes and attributes present within playlists, while still leaving the possibility of the original intention. By leveraging Spotify's API to gather song attributes and the Genius API for lyrics and sentiment analysis, I employ Positive-Unlabeled (PU) learning to handle the absence of explicitly disliked songs. This report details the methodologies, challenges, and results of my approach to refining music recommendations.

1\. Introduction
================

**Objective**: The project began with the goal of recommending songs based on a user's liked songs from Spotify. Due to the nuances in personal preferences in songs and limitations in available data, the focus shifted to tailoring personalized playlists, allowing for the model to leverage the thematic coherence of playlists, thus improving the quality of recommendations.

**Motivation**: Spotify has a notorious presence of having poor recommendations leaving users wanting a better system for song recommendations.

**Challenges**: Handling positive-unlabeled learning due to the lack of disliked songs and ensuring model success while avoiding both over fitting and proper use of reliable negatives.

2\. Data Collection
===================

**Spotify API**: Calls to Spotify's "tracks", "audio-features", and "playlist/<playlist-id>/tracks" endpoints are used to fetch data on songs within a playlist. The data collects song attributes such as danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_min, and time_signature, popularity, track_id, title, and artists.

**Genius API**: Extracted lyrics for sentiment analysis. Genius does not directly support the fetching of lyrics, so a combination of API requests as well as HTML. To accomplish this first a call is made to the API's search endpoint using the song title and artist as search parameters. The API responds with metadata including the path to the lyrics page. Using the path, a full URL is constructed for the page and an HTTP request is made to fetch the HTML content. Using BeautifulSoup, the function parses the HTML to find the 'div' elements that contain the lyrics.

3\. Data Preprocessing
======================

**Lyric Cleansing**: Using the Regular Expressions library (re), Genius annotations such as [Chorus: Pharell Williams] were removed as they add no benefit to sentiment and could even alter the analysis negatively.

**Feature Selection**: Features used in training in this project includes the 10 audio features, popularity, and release date from Spotify as well as "pos" and "neg" features from VADER sentiment analysis. Features "mode" and "time_signature" were removed as there was minimal correlation between the features and any sets during testing. Additionally removing them lowered model complexity. "loudness" showed potential as a feature, however, even among genres considered "loud" such as rock, loudness proved inconsistent in its value. On top of this fact, Spotify, by default, uses a normalizer to make all songs the same volume therefore removing little chance that a user would make selection based on this. Lastly, the "neu" and "compound" features from VADER were removed for two different reasons. "Neu" (Neutral) represents filler words such as"the", "a", and any words that provide no value which the model can make use of. "compound" is a combination of the values "pos" and "neu" so it was removed as the information is already present in these other features in a more detailed manner.

**Feature Engineering**: "years_since_release" was created from Spotify's release_date (string) to create a usable float for the model.

**Normalization**: The following features with a range of [0,1] were normalized with the cube root due to right-skewed data: "instrumentalness", "acousticness". "liveness", "pos", "neg", "speechiness". Cube root was selected initially to treat the data similarly to "compound" which had a range of [-1,1] and cube root allowed for sign preservation. Cube root was left in place, even after the removal of "compound"  as it nicely distributed the values across the range. "danceability" and "energy" both have ranges of [0, 1] but were left-skewed and so the features were cubed to normalize and redistribute the data nicely.

**Scaling**: "duration_ms" from Spotify was scaled to represent minutes, decreasing values by a factor of 60,000 without ruining the distribution.

**Dataset Structure**: The backbone of this project was Pandas' dataframes. These dataframes allowed for the efficient lookup and storage of data as well as being compatible with other libraries such as Sklearn and XGBoost.

4\. Methodology
===============

**Sentiment Analysis**: VADER sentiment analysis was selected for use on song lyrics as it was trained on social media which made it perfect for analyzing lyrics containing euphemisms or slang words which are common across most genres but also vary in implementation on Genius. Additionally, VADER is well-suited for dealing with misspelled or shorthand lyrics such as "dancing" being written as "dancin" which is extremely popular in modern music.

**Model Selection**: Three models were evaluated for selection: Sklearn's LogisticRegression, Sklearn's RandomForrestClassifier, and XGBoost's XGBClassifier. Decision trees were selected early on to deal with the once imbalanced data after finding reliable negatives. Ultimately XGBoost was selected over RandomForrestClassifer as it proved to be slightly more reliable.

**Handling Imbalanced Data**: Initially imbalanced data was dealt solely by using decision trees which perform well even with imbalanced data. However, due to the random nature of the level of the imbalance, ImbLearn's SMOTE was used to over sample the minority class and balance the two. Over sampling was chosen over under sampling as the model expects little data to begin with and so losing data is not an option.

**Model Training**: For Positive Unlabeled Learning, several steps must be taken to ensure the model is properly trained. These steps are as follows:

1.  First a threshold at which to select reliable negatives from is set. For this project 0.1, or 10% probability of being positive, was the threshold for negatives

2.  XGBoost is initialized and trained on the data where 1 is the positive class and the unknown data is labeled and treated as 0s, or the negative class.

3.  Using this trained model and the predict_proba() method from XGBoost, probabilities of all the data being positive is generated

4.  From this list, values with a 10% or less chance of being positive are selected as reliable negatives. These values are treated as the negative class.

5.  From here, the positive data from the start and the reliable negatives are rebalanced using oversampling. Any data not in the positive class or selected as a reliable negative is ignored.

6.  Using the new, balanced, positive and negative data, XGBoost is retrained on this model that presented.

For positive-unlabeled learning the reliable negative threshold was selected as 0.10, or 10%, to ensure that only songs that are as far as possible from being positive were selected. Incorrectly selecting songs as negative could cause the model to treat potential positive songs as negative moving forward.

5\. Model Evaluation
====================

**F1-Score**: The primary measure of model performance was F1-Score as it punishes false positives and false negatives, or recommending bad songs and not recommending good songs.

**Accuracy**: Accuracy was used as an additional measure of performance as it is more intuitive to users as it is simply a measure of songs the model was correct on.

**Validation Set**: Validation of the model was difficult due to the unknown truth values of much of the data. To subvert this, a validation set was manually created out of the dataframe for "Hardest Rap Songs" and "Biggest Songs of All Time Top500" playlists. This set was partially flawed as the definition of a "hard" rap song can be debated so this was only used during validation to compare the results of two models. For songs that were considered borderline, they were labeled as positive to encourage recommending them rather than them going unseen by the user. 

6\. Results
===========

**Test-Sets**: To eliminate bias in the models three separate test sets were created from various sources. To further see how the model performed under different circumstances, each test set had differing attributes. These attributes were a genre specific playlist, an expansive, loosely organized playlist, and a mood specific playlist. Those sets are as follows.

1.  The first test set was created by a third-party. This included their personal 63 song playlist consisting primarily of EDM and dance music as the positive, target playlist as well as the top 500 songs of all time playlist found on spotify as the unknown playlist. For this dataset a test set of 10% or 58 songs was used as the test set songs had to be manually labeled by the third-party. One important thing to note for this dataset was the unknown data contained virtually no positive instances and so this shows the ability of the model to function at the extremes. With this the model achieved the following metrics.

       **F1-Score**: 0.9231

       **Accuracy**: 0.9825

2.  The second test set was created by a different third-party than the first. This playlist consisted of 60 songs from a playlist they made for the purposes of this project consisted largely of pre-2000's dancing music with a sizable minority being scattered across genres such as pop rock, slap rap, and synthwave. Like test set 1, this dataset saw a test set of 10% or 56 songs as it also had to be manually labeled by the third-party. This set was also an interesting view into the model's abilities as the songs in the playlist crossed many genres and had very expansive characteristics. With all of this, the model achieved the following metrics.

       **F1-Score**: 0.0625;

       **Accuracy**: 0.4545

3.  The final test set was curated using my personal playlists in a manner that attempted to remove bias and utilize the model exactly as intended. The target playlist was an already existing playlist meant for the gym. The larger unknown playlist was a comprehensive playlist of nearly all songs I like, including those within the target playlist. To ensure the unknown playlist contained some unknown songs, a random subset of the target playlist was removed, but their truth value was stored for evaluation. This truth value was used to make it so there was no manual checking required of the test set, eliminating user bias.

       **F1-Score**: 0.500;

       **Accuracy**: 0.8692

7\. Discussion
==============

**Insights**:

-   **Complexity of Positive-Unlabeled (PU) Learning**: One of the most significant challenges from this project was the inherent complexity of implementing PU learning for playlist curation. This challenge stemmed from the lack of explicitly labeled negative examples, which required careful handling to avoid bias in the model. The process of identifying reliable negative and rebalancing the dataset was crucial in achieving a reasonable model performance. Additionally, this approach highlighted the importance of carefully setting thresholds for identifying reliable negatives, as even slight variations in this threshold could significantly impact the model outcomes.

-   **Significance of Feature Selection**: The project also underscored the critical role of feature selection in improving model accuracy and reducing complexity. Features such as 'mode' and 'time_signature' were removed after determining they were removed after determining they added minimal value, simplifying the model and improving its performance. Similarly, decisions around which sentiment features to include demonstrated the need for targeted feature selection to tailor to the specific nuances of music recommendation.

**Challenges**: 

-   **Data Imbalance**: One of the primary challenges faced was managing the imbalance between the positive and unlabeled classes. Initially this was solved by simply using a decision tree, however in some extreme scenarios, this was not sufficient. Implementing SMOTE oversampling helped mitigate this issue.

-   **Sentiment Analysis**: Another significant challenge was accurately assessing sentiment from song lyrics. Given the prevalence of slang, euphemisms, and unconventional language in lyrics, traditional sentiment analysis tools struggled to provide accurate sentiment scores. While VADER was selected for its suitability to social media-style text, there were still instances where it misinterpreted the context or tone of the lyrics. Future iterations may explore more advanced NLP techniques or custom models trained specifically on song lyrics to overcome this limitation.

**Limitations**: 

-   **Subjectivity in Evaluatio**n: One of the limitations of this project is the inherent subjectivity in evaluating the model's success. Music preferences are highly personal, and there is no definitive way to determine if a song fits within a particular theme or genre. The test sets were carefully curated, but human judgment was still required, which introduces potential bias. Songs that would fit into the playlist one day, may not be selected for the playlist on the next and this is simply unavoidable. Furthermore, the F1-Score and accuracy metrics provide only a partial view of the model's performance; they may not fully capture the model's ability to recommend songs that align with the user's taste. Songs may be a good recommendation but the user may not like that specific song or artist, but this song is treated the same as bad recommendations.

-   **Generalizability**: The model's performance might be limited by the specific playlists and user preferences it was trained on. While efforts were made to create diverse test sets, the model may not generalize well to playlists with very different themes or genres. Additionally, the reliance on sentiment analysis from the Genius API introduces a dependency on the quality and accuracy of available lyrics, which could vary between songs.

**Discussion of Results**: The three test sets used show varying levels of success all for differing reasons. Test set 1 displayed the models exceptional abilities to curate a playlist based on genre utilizing the characteristics of the audio features. For usage in this type of playlist creation, the model has excelled and users certainly would be satisfied with the recommendation based on this. Test set 2 is partially disappointing however, this was to be mostly expected due to the wide coverage the playlist hoped to accomplish which suffered heavily from a limited input. This set would likely be one of the most difficult to achieve fair F1-Scores for even the best of models. Even so, the model performing worse than the odds of a coin-flip is a let down. The most important test set of all, test set 3, performed moderately and provides hope for improvements of the model. This test set utilized the exact intended use of the model and provided promising results, an accuracy of 0.86, that most users would be happy to have as a way to presort songs.

8\. Conclusion
==============

**Summary**: This project set out to enhance music recommendation systems by creating a personalized playlist curation model using Positive-Unlabeled (PU) learning. By leveraging Spotify's song attributes and sentiment analysis from the Genius API, the model aimed to identify songs that fit a user's personal taste without the need for explicit negative examples. The project successfully navigated the challenges of imbalanced data and the complexities of feature selection, resulting in a model that achieved reasonable accuracy and provided insightful recommendations.

**Future Work**: Future iterations of this project could explore several avenues for improvement. Incorporating additional features, such as user-specific listening habits or more granular genre classifications, could enhance the model's ability to tailor recommendations. Additionally, experimenting with alternative algorithms or ensemble methods could further refine the model's accuracy and robustness. Another promising area for future work is the development of custom sentiment analysis tools specifically trained on song lyrics, which could better capture the nuances of musical language. Finally, the ability to have access to how often a user skips a song could allow for a similar model to be on par with that of Spotify by turning this into a fully supervised model.

**Impact**: The potential impact of this project lies in its ability to significantly enhance user experience in music recommendation systems. By moving beyond traditional collaborative filtering methods and embracing a more nuanced approach to playlist curation, this model offers a more personalized and context-aware recommendation system. This could lead to higher user satisfaction and engagement for streaming platforms, as the recommendations would better align with individual preferences and tastes.

9\. References
==============

1.  **Spotify API Documentation**. (n.d.). Retrieved from[  https://developer.spotify.com/documentation/web-api/](https://developer.spotify.com/documentation/web-api/)

-   Used to collect song attributes such as "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature", "popularity", "track_id", "title", and "artists".

2.  **Genius API Documentation**. (n.d.). Retrieved from https://docs.genius.com/

-   Used for extracting lyrics for sentiment analysis.

3. ** VADER Sentiment Analysis Tool**. (Hutto, C., & Gilbert, E.). (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Retrieved from[  https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment)

-   Applied for sentiment analysis of song lyrics due to its effectiveness with informal language and social media text.

4.  **Scikit-Learn Documentation**. (Pedregosa, F., et al.). (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830. Retrieved from https://scikit-learn.org/stable/documentation.html

-   Used for implementing Logistic Regression and Random Forest Classifier.

5.  **XGBoost Documentation**. (Chen, T., & Guestrin, C.). (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Retrieved from[  https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)

-   Employed for training the final model due to its performance with imbalanced data.

6. **SMOTE (Synthetic Minority Over-sampling Technique)**. (Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.). (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.

-   Used to handle imbalanced data by oversampling the minority class.

7. **Pandas Documentation**. (McKinney, W.). (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51-56. Retrieved from https://pandas.pydata.org/

-   Used for data manipulation and analysis, providing the backbone of the project's data handling.

8. **Matplotlib Documentation**. (Hunter, J. D.). (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95. Retrieved from https://matplotlib.org/stable/contents.html

-   Used for plotting distributions and visualizing the data.

9. **Imbalanced-learn Documentation**. (Lemaître, G., Nogueira, F., & Aridas, C. K.). (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning. Journal of Machine Learning Research, 18(17), 1-5. Retrieved from https://imbalanced-learn.org/stable/

-   Used for implementing SMOTE and handling imbalanced data.

10. **Python Documentation**. (Van Rossum, G., & Drake, F. L.). (2001). Python Reference Manual. PythonLabs. Retrieved from[  https://docs.python.org/3/](https://docs.python.org/3/)

-   The core programming language used for all aspects of the project.
