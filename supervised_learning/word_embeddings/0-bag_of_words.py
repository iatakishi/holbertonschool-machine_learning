#!/usr/bin/env python3
"""
Bag of Words Module
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list): A list of sentences to analyze.
        vocab (list): A list of the vocabulary words to use for the analysis.
                      If None, all words within sentences should be used.

    Returns:
        embeddings: A numpy.ndarray of shape (s, f) containing the embeddings.
                    - s is the number of sentences.
                    - f is the number of features analyzed.
        features: A numpy.ndarray list of the features used for embeddings.
    """
    extracted_sentences = []

    for sentence in sentences:
        # Convert sentence to lowercase
        s = sentence.lower()

        # Remove possessive "'s" to avoid parsing lone "s" as a standalone word
        s = re.sub(r"'s\b", "", s)

        # Extract all alphanumeric words using word boundaries
        words = re.findall(r'\b\w+\b', s)
        extracted_sentences.append(words)

    # Build a sorted vocabulary if one is not provided
    if vocab is None:
        vocab_set = set()
        for words in extracted_sentences:
            vocab_set.update(words)
        vocab = sorted(list(vocab_set))

    # Ensure vocab is a list (handles tuple inputs seamlessly)
    vocab = list(vocab)

    # Initialize the embeddings matrix with zeros
    s_len = len(sentences)
    f_len = len(vocab)
    embeddings = np.zeros((s_len, f_len), dtype=int)

    # Populate the embeddings matrix by counting occurrences
    for i, words in enumerate(extracted_sentences):
        for word in words:
            if word in vocab:
                # Increment the count for the given word's feature index
                embeddings[i, vocab.index(word)] += 1

    return embeddings, np.array(vocab)
