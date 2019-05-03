from __future__ import print_function   # Ensures compatibility in Python versions 3.x and 2.x
from time import time   # Used for measuring time elapsed for a process

from sklearn.feature_extraction.text import TfidfVectorizer    # TfidVectorizer is used for converting a collection of raw documents to a matrix of TF-IDF features
from sklearn.decomposition import NMF    # Non-negative Matrix Factorization
from sklearn.datasets import fetch_20newsgroups # Used for loading the 20 newsgroup dataset

n_top_words = 5

def print_top_words(model, feature_names, n_top_words):
    message = ""
    for topic_idx, topic in enumerate(model.components_):
        message += "T%d: " % (topic_idx + 1)
        message += " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        message += "<br/>"
    return message

def get_nmf():

    print("Loading dataset...")
    t0 = time()
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data
    print("done in %0.3fs." % (time() - t0))

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=2000, min_df=10, stop_words='english')
    # max_features: builds a vocabulary that only consider the top 2000 ordered by term frequency across the corpus.
    # min_df: ignores terms that have a document frequency strictly lower than the given threshold
    # stop_words: words which are filtered out before or after processing of natural language text. (ex. "and", "the", "him")

    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(data_samples)
    # fit_transform() = fit() + transform() = Learn vocabulary and idf, return term-document matrix.
    print("done in %0.3fs." % (time() - t0))
    print()

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features")
    t0 = time()
    nmf = NMF(n_components=10, solver="mu").fit(tfidf)  # Learn a NMF model for TF-IDF
    # n_components: number of topics
    # solver: Numerical solver to use. 'cd'=Coordinate Descent solver; 'mu'=Multiplicative Update solver
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    return print_top_words(nmf, tfidf_feature_names, n_top_words)