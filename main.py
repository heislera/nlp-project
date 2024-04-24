import os
import string
import warnings

import gensim
import nltk
import pandas as pd
import re
import spacy
import contractions
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize

pd.set_option('display.max_columns', None)
nltk.download('punkt')





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
        # if field != 'genres':
        movie_data[field] = movie_data[field].apply(lambda x: re.sub("[^a-z0-9' ]", "", str(x)))

    # split contractions for movie desc
    # movie_data = movie_data['movie_info'].apply(lambda x: ' '.join([decontract(word) for word in x.split()]))


def clean_review_data(critic_reviews):
    # remove nulls from review content
    critic_reviews = critic_reviews[critic_reviews['review_content'].notnull()]

    # set review content to lowercase
    critic_reviews.loc[:, 'review_content'] = critic_reviews['review_content'].apply(lambda x: str(x).lower())
    critic_reviews.loc[:, 'review_content'] = critic_reviews["review_content"].apply(remove_punctuation)
    # split contractions, this can be optional
    # this is really slow, maybe we just run this once and save the data as a separate column
    # critic_reviews['review_content'] = remove_contractions_from_reviews(critic_reviews)

    return critic_reviews




def decontract(word):
    return contractions.fix(word)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

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


def get_reviews(critic_reviews, movie_link):
    return critic_reviews[critic_reviews["rotten_tomatoes_link"] == movie_link]

def create_training_data(critic_reviews, movie_data, features=None):
    # this creates a file with the new cleaned and merged data column
    # this column is what is fed into models for training for similarity probably
    # if features contains True as the last attribute, all movie reviews will be included
    """
    Index(['rotten_tomatoes_link', 'movie_title', 'movie_info',
       'critics_consensus', 'content_rating', 'genres', 'directors', 'authors',
       'actors', 'original_release_date', 'streaming_release_date', 'runtime',
       'production_company', 'tomatometer_status', 'tomatometer_rating',
       'tomatometer_count', 'audience_status', 'audience_rating',
       'audience_count', 'tomatometer_top_critics_count',
       'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count'],
      dtype='object')

    """

    use_reviews = False

    if features[-1]:
        use_reviews = True
        features = features[:-1]

    df_name = "_".join(features)

    if use_reviews:
        df_name += "_WITH_REVIEWS"

    if os.path.exists(f"data/{df_name}.csv"):
        print("data already exists")
        return

    print("making movie documents")
    # this modifies the movie_data dataframe to have a new "merged_features" column.
    merge_features(movie_data, features)

    if use_reviews:
        # iterate over all movies, grab all reviews for that movie.
        # merge those reviews into a single string, then append it at then end of the merged features
        for index, row in movie_data.iterrows():
            reviews = get_reviews(critic_reviews, row.rotten_tomatoes_link)
            merged_reviews = " ".join(str(val) for val in reviews.review_content)
            movie_data.at[index, "merged_features"] += " " + merged_reviews

            if index % 100 == 0:
                print(index/movie_data.shape[0])

    movie_data.to_csv(f"data/{df_name}.csv")
    print("finished making movie documents")

def train_model(features, vector_size=50, epochs=20):
    use_reviews = False

    if features[-1]:
        use_reviews = True
        features = features[:-1]

    df_name = "_".join(features)

    if use_reviews:
        df_name += "_WITH_REVIEWS"

    training_df = pd.read_csv(f"data/{df_name}.csv")

    tagged_data = []
    print("begin making tagged data for model training")
    for index, row in training_df.iterrows():
        tagged_doc = TaggedDocument(words=word_tokenize(row.merged_features),
                                    tags=[str(row.movie_title)])
        tagged_data.append(tagged_doc)
    print("finish making tagged data for model training")

    print("begin training")
    # docs for this are here https://radimrehurek.com/gensim/models/doc2vec.html
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)

    model.build_vocab(tagged_data)

    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)
    model.save("model.model")
    print("finish training")

    return model
def generate_model(critic_reviews, movie_data, features):
    create_training_data(critic_reviews, movie_data, features)
    model = train_model(features)

    return model

def main() -> None:
    critic_reviews, movie_data = load_data()

    # for some reason this needs to be assigned
    critic_reviews = clean_review_data(critic_reviews)
    clean_movie_data(movie_data)

    # True at the end of this list tells the method that creates documents from our data to include review data
    movie_data_features = ["movie_title", "genres", "directors", "actors", "movie_info", "critics_consensus", True]

    # model = generate_model(critic_reviews, movie_data, movie_data_features)

    training_df = pd.read_csv(f"data/movie_title_genres_directors_actors_movie_info_critics_consensus_WITH_REVIEWS.csv")
    model = Doc2Vec.load("model.model")

    # below this is just testing the model

    test_doc = training_df.at[0, "merged_features"]
    # print(movie_data[movie_data["rotten_tomatoes_link"] == critic_reviews.at[0, "rotten_tomatoes_link"]]["movie_title"])
    # print(test_doc)

    # this is how we calculate similarity for an input doc
    tokenized_doc = word_tokenize(test_doc.lower())
    inferred_vector = model.infer_vector(tokenized_doc)
    similar_docs = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # the test doc is the very first movie, removing this check will similarity to all movies.
    # adjusting similar_docs to similar_docs[:10] will only print 10
    for doc_id, similarity in similar_docs:
        if doc_id == "percy jackson  the olympians the lightning thief":
            print("Document ID:", doc_id, "Similarity Score:", similarity)



    # old code

    # desc = input("Describe your movie, then press Enter: ").lower()
    # desc1 = re.sub("[^a-z0-9' ]", "", desc)
    #
    # with warnings.catch_warnings(action="ignore"):
    #     compute_spacy_similarity(movie_data, desc1)
    #
    # similarity_minimum = 0.6
    # temp1 = movie_data[movie_data["similarity"] >= similarity_minimum]
    # temp2 = temp1.sort_values(by=["similarity"], ascending=False)
    # print("the top movies with a similarity greater than 0.6:", temp2[["movie_title", "similarity"]].head(10))
    #
    # input("\nPress Enter to Close.")



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
