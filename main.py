import os
import random
import string
import time
from collections import defaultdict
from nltk.corpus import stopwords

import gensim
import nltk
import pandas as pd
import re
import contractions
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import word_tokenize
import yake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

pd.set_option('display.max_columns', None)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

global movie_titles_id


def load_data():
    global movie_titles_ids
    critic_reviews = pd.read_csv("data/rotten_tomatoes_critic_reviews.csv")
    movie_data = pd.read_csv("data/rotten_tomatoes_movies.csv")

    # This is done so we can access the unedited movie titles later (to display to user)
    # Extract movie titles and links
    movie_titles = movie_data["movie_title"]
    rotten_tomatoes_links = movie_data["rotten_tomatoes_link"]
    # Combine titles and links into a DataFrame
    movie_titles_ids = pd.DataFrame({"movie_title": movie_titles, "rotten_tomatoes_link": rotten_tomatoes_links})

    return critic_reviews, movie_data


def clean_movie_data(movie_data):
    fields = ['movie_title', 'movie_info', 'content_rating', 'genres', 'directors', 'authors', 'actors',
              'production_company']
    for field in fields:
        # movie_data = movie_data[movie_data[field].notnull()] # needed?

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


def get_movie_title(movie_link):  # helper method to get unedited movie title from doc_id
    return movie_titles_ids[movie_titles_ids["rotten_tomatoes_link"] == movie_link]["movie_title"].iloc[0]


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
                print(index / movie_data.shape[0])

    movie_data.to_csv(f"data/{df_name}.csv")
    print("finished making movie documents")


def train_similarity_model(features, vector_size=100, epochs=20):
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
                                    tags=[row.rotten_tomatoes_link])
        tagged_data.append(tagged_doc)
    print("finish making tagged data for model training")

    print("begin training")
    # docs for this are here https://radimrehurek.com/gensim/models/doc2vec.html
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, epochs=epochs)

    model.build_vocab(tagged_data)

    model.train(tagged_data, total_examples=model.corpus_count, epochs=epochs)
    model.save("model.model")
    print("finish training")

    return model


def generate_model(critic_reviews, movie_data, features):
    create_training_data(critic_reviews, movie_data, features)
    model = train_similarity_model(features)

    return model

def train_random_forest_model(critic_reviews):
    """
    Requires the following:
    - A list of reviews
    - labels for each reviews
    - a list of keywords
    """
    reviews = critic_reviews["review_content"]
    keywords = critic_reviews["keywords"]

    # Flatten the list of lists of keywords to a single list of strings
    keywords = [keyword for sublist in keywords for keyword in sublist]

    # Convert keywords to a set to remove duplicates, then put it back in a list
    keywords = list(set(keywords))

    labels = critic_reviews["label"]

    # todo maybe binary should be true, it's mentioned in the chatGPT thing im using as reference to this
    vectorizer = CountVectorizer(vocabulary=keywords, binary=False)
    X = vectorizer.fit_transform(reviews)

    rfc = RandomForestClassifier(n_estimators=3)
    rfc.fit(X, labels)

    return rfc


def generate_plot(keywords, rfc):
    keywords = [keyword for sublist in keywords for keyword in sublist]

    # Convert keywords to a set to remove duplicates, then put it back in a list
    keywords = list(set(keywords))

    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
    d = {"importances": importances, "std": std}
    forest_importances = pd.DataFrame(d, index=keywords)
    forest_importances.sort_values(by="importances", ascending=False, inplace=True)
    forest_importances = forest_importances[:10]
    fig, ax = plt.subplots()
    forest_importances["importances"].plot.bar(yerr=forest_importances["std"], ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def generate_wordcloud(keyword_importance):
    # Convert list of tuples to dictionary
    keyword_importance_dict = dict(keyword_importance)

    wordcloud = WordCloud(width=800, height=400, background_color="black")
    wordcloud.generate_from_frequencies(keyword_importance_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def extract_important_features(model, keywords):
    # Flatten the list of lists of keywords to a single list of strings
    keywords = [keyword for sublist in keywords for keyword in sublist]

    # Convert keywords to a set to remove duplicates, then put it back in a list
    keywords = list(set(keywords))
    importances = model.feature_importances_

    # print importances next to keywords
    keyword_importance = dict(zip(keywords, importances))

    sorted_keywords = sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)

    return sorted_keywords



def merge_features(dataframe, features):
    """
    this method takes a dataframe and a list of its features, and creates a new feature called "merged_features"
    this features is simply a concatenated string of all non-nan features provided
    """
    dataframe['merged_features'] = dataframe[features].apply(
        lambda row: ' '.join(str(val) for val in row if pd.notna(val)), axis=1)


def merge_movie_review_features(percentage, critic_reviews):
    # Takes a movie link and percentage of reviews to use, then returns string of reviews in random order
    sample_size = int(len(critic_reviews) * percentage / 100)
    rand_reviews = random.sample(critic_reviews["review_content"].toList(), sample_size)

    return rand_reviews

def extract_review_keywords(reviews, num_keywords=3):
    data = []

    for review, label in zip(reviews["review_content"], reviews["review_type"]):
        yake_kw = yake.KeywordExtractor()
        keywords = yake_kw.extract_keywords(review)

        # Filter out single-word keywords - this is done because most of the single-word keywords
        # aren't very helpful (ex. 'lightning', 'fantasy' were returned for percy jackson as important keywords)
        keywords = [keyword[0] for keyword in keywords if " " not in keyword[0]]

        # Sort keywords based on their scores in descending order
        keywords_sorted = sorted(keywords, key=lambda x: x[1], reverse=True)

        # Get top n keyword phrases and append to dataframe
        top_n_keywords = keywords_sorted[:num_keywords]
        data.append({'label': label, 'review_content': str(review), 'keywords': top_n_keywords})

    review_df = pd.DataFrame(data)

    return review_df


def co_occurrence_analysis(text_series, keyword, window_size=3):
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    stop_words = set(stopwords.words('english'))

    for text in text_series:
        # Tokenize the text
        tokens = nltk.word_tokenize(text.lower())

        # Create a co-occurrence matrix for each text in the series
        for i in range(len(tokens)):
            # Find keyword in token list
            if tokens[i] == keyword:
                # Calculate window index around the keyword
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                    # Ignore the keyword and stop words
                    if i != j and tokens[j] not in stop_words:
                        co_occurrence_matrix[keyword][tokens[j]] += 1

    # Sort the co-occurrence dictionary by counts in descending order
    sorted_co_occurrences = dict(sorted(co_occurrence_matrix[keyword].items(), key=lambda x: x[1], reverse=True))
    return sorted_co_occurrences

def main() -> None:
    critic_reviews, movie_data = load_data()

    # for some reason this needs to be assigned
    critic_reviews = clean_review_data(critic_reviews)
    clean_movie_data(movie_data)

    # True at the end of this list tells the method that creates documents from our data to include review data
    movie_data_features = ["movie_title", "genres", "directors", "actors", "movie_info", "critics_consensus", True]

    # check if the model exists, generate one if it doesn't
    if os.path.exists("model.model"):
        model = Doc2Vec.load("model.model")
    else:
        model = generate_model(critic_reviews, movie_data, movie_data_features)
        model.save("model.model")

    training_df = pd.read_csv(f"data/movie_title_genres_directors_actors_movie_info_critics_consensus_WITH_REVIEWS.csv")

    # below this is just testing the model

    test_doc = training_df.at[0, "merged_features"]

    # this is how we calculate similarity for an input doc
    tokenized_doc = word_tokenize(test_doc.lower())
    inferred_vector = model.infer_vector(tokenized_doc)
    similar_docs = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    review_data = []

    # for now, the keywords are extracted from only the top 5 movies (similarity scores)
    num_similar_movies = 5
    for doc_id, similarity in similar_docs[:num_similar_movies]:
        movie_title = get_movie_title(doc_id)
        print("Similar movie: ", movie_title, "\tSimilarity Score: ", similarity)
        spec_reviews = get_reviews(critic_reviews, str(doc_id))
        keyword_df = extract_review_keywords(spec_reviews)
        review_data.append(keyword_df)

    # Concatenate the dataframes generated in the loop above
    review_df = pd.concat(review_data, ignore_index=True)

    rfm = train_random_forest_model(review_df)

    sorted_keywords = extract_important_features(rfm, review_df["keywords"])

    # generate_plot(review_df["keywords"], rfm)
    # generate_wordcloud(sorted_keywords)

    # print co occurrences
    keyword_count = 5
    co_word_count = 10
    for keyword_tuple in sorted_keywords[:keyword_count]:
        word = keyword_tuple[0]
        co_occurrence = co_occurrence_analysis(review_df["review_content"], word)
        top_words = [(k,v) for k, v in co_occurrence.items()][:co_word_count]
        print(keyword_tuple, top_words)



if __name__ == '__main__':
    main()
