#!/usr/bin/env python3
"""
TF-IDF Module
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of the vocabulary words to use for the analysis.
                      If None, all words within sentences should be used.

    Returns:
        embeddings: A numpy.ndarray of shape (s, f) containing the embeddings.
                    - s is the number of sentences in sentences.
                    - f is the number of features analyzed.
        features: A numpy.ndarray list of the features used for embeddings.
    """
    # Initialize the vectorizer with the specified vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences into a TF-IDF matrix
    X = vectorizer.fit_transform(sentences)

    # Extract feature names safely depending on the scikit-learn version
    try:
        features = vectorizer.get_feature_names_out()
    except AttributeError:
        features = vectorizer.get_feature_names()

    return X.toarray(), np.array(features)
