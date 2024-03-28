import pandas as pd

def load_data():
    critic_reviews = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
    movie_data = pd.read_csv("data/rotten_tomatoes_movies.csv")
    return critic_reviews, movie_data

def main() -> None:
    critic_reviews, movie_data = load_data()

    # remove nulls from review content
    critic_reviews = critic_reviews[critic_reviews['review_content'].notnull()]

    # set review content to lowercase
    critic_reviews['review_content'] = critic_reviews['review_content'].apply(lambda x: str(x).lower())

    temp = critic_reviews['review_content'].apply(lambda x: x.split())



if __name__ == '__main__':
    main()
