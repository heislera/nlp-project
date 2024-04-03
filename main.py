import warnings

import pandas as pd
import re
import spacy
import contractions
import numpy as np



pd.set_option('display.max_columns', 30)




def load_data():
    critic_reviews = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
    movie_data = pd.read_csv("data/rotten_tomatoes_movies.csv")
    return critic_reviews, movie_data


def clean_movie_data(movie_data):
    fields = ['movie_title', 'movie_info', 'content_rating', 'genres', 'directors', 'authors', 'actors', 'production_company']
    for field in fields:
        #movie_data = movie_data[movie_data[field].notnull()] # needed?

        # set content to lowercase
        movie_data[field] = movie_data[field].apply(lambda x: str(x).lower())

        # remove all punctuation except for apostrophes (for contractions)
        if field != 'genres':
            movie_data[field] = movie_data[field].apply(lambda x: re.sub("[^a-z0-9' ]", "", str(x)))

    # split contractions for movie desc
    # movie_data = movie_data['movie_info'].apply(lambda x: ' '.join([decontract(word) for word in x.split()]))


def clean_review_data(critic_reviews):
    # remove nulls from review content
    critic_reviews = critic_reviews[critic_reviews['review_content'].notnull()]

    # set review content to lowercase
    critic_reviews['review_content'] = critic_reviews['review_content'].apply(lambda x: str(x).lower())

    # split contractions, this can be optional
    # this is really slow, maybe we just run this once and save the data as a separate column
    # critic_reviews['review_content'] = remove_contractions_from_reviews(critic_reviews)
    pass


def decontract(word):
    return contractions.fix(word)


def remove_contractions_from_reviews(critic_reviews):
    """
    Note: This may cause issues for certain contractions like ain't which may split into multiple pairs of words.
    Just something to keep in mind for this in the future.
    """
    return critic_reviews['review_content'].apply(lambda x: ' '.join([decontract(word) for word in x.split()]))


"""
movies.csv columns

Index(['rotten_tomatoes_link', 'movie_title', 'movie_info',
       'critics_consensus', 'content_rating', 'genres', 'directors', 'authors',
       'actors', 'original_release_date', 'streaming_release_date', 'runtime',
       'production_company', 'tomatometer_status', 'tomatometer_rating',
       'tomatometer_count', 'audience_status', 'audience_rating',
       'audience_count', 'tomatometer_top_critics_count',
       'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count'],
      dtype='object')

    
possible feature combinations to try (from this list of features at least)
we can also include permutations of this data, but i'm not sure if that change anything

['movie_info', 'genres', 'directors', 'authors']
['movie_info', 'genres', 'directors']
['movie_info', 'genres', 'directors', 'critics_consensus']

if we allow specific titles to be said (my request to friends for some test data likely wont have this)

['movie_title', 'movie_info', 'genres', 'directors', 'authors', ]
['movie_title', 'movie_info', 'genres', 'directors']
['movie_title', 'movie_info', 'genres', 'directors', 'critics_consensus']


we may also dump reviews into this text blob:
- all reviews
- all positive reviews
- all negative reviews
- random sample of reviews

"""


def main() -> None:
    critic_reviews, movie_data = load_data()

    clean_review_data(critic_reviews)
    clean_movie_data(movie_data)

    merge_features(movie_data, ['genres', 'movie_info'])

    desc = input("Describe your movie, then press Enter: ").lower()
    desc1 = re.sub("[^a-z0-9' ]", "", desc)

    with warnings.catch_warnings(action="ignore"):
        compute_spacy_similarity(movie_data, desc1)

    similarity_minimum = 0.6
    temp1 = movie_data[movie_data["similarity"] >= similarity_minimum]
    temp2 = temp1.sort_values(by=["similarity"], ascending=False)
    print(temp2["movie_title"], temp2["similarity"])


def merge_features(dataframe, features):
    """
    this method takes a dataframe and a list of it's features, and creates a new feature called "merged_features"
    this features is simply a concatenated string of all non-nan features provided
    """
    dataframe['merged_features'] = dataframe[features].apply(lambda row: ' '.join(str(val) for val in row if pd.notna(val)), axis=1)



def compute_spacy_similarity(dataframe, input):
    # testing some stuff
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    dataframe['similarity'] = 0.0
    for i in range(dataframe["merged_features"].size):
        doc2 = nlp(dataframe["merged_features"].iloc[i])
        dataframe.at[i, "similarity"] = doc.similarity(doc2)

        if i % 300 == 0:
            print(i/dataframe["merged_features"].size)



if __name__ == '__main__':
    main()
