import pandas as pd
import re

import contractions

def load_data():
    critic_reviews = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
    movie_data = pd.read_csv("data/rotten_tomatoes_movies.csv")
    return critic_reviews, movie_data

def decontract(word):
    return contractions.fix(word)

def remove_contractions_from_reviews(critic_reviews):
    """
    Note: This may cause issues for certain contractions like ain't which may split into multiple pairs of words.
    Just something to keep in mind for this in the future.
    """
    return critic_reviews['review_content'].apply(lambda x: ' '.join([decontract(word) for word in x.split()]))

def main() -> None:
    critic_reviews, movie_data = load_data()

    # remove nulls from review content
    critic_reviews = critic_reviews[critic_reviews['review_content'].notnull()]

    # set review content to lowercase
    critic_reviews['review_content'] = critic_reviews['review_content'].apply(lambda x: str(x).lower())

    # split contractions
    critic_reviews['review_content'] = remove_contractions_from_reviews(critic_reviews)


    desc = input("Describe your movie, then press Enter: ").lower()
    desc1 = re.sub("[^a-z0-9' ]", "", desc).split(" ")



if __name__ == '__main__':
    main()
