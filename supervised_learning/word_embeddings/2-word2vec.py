#!/usr/bin/env python3
"""
Word2Vec Model Module
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model.

    Args:
        sentences (list): A list of sentences to be trained on.
        vector_size (int): The dimensionality of the embedding layer.
        min_count (int): The minimum number of occurrences of a word for use.
        window (int): The maximum distance between the current and predicted
                      word within a sentence.
        negative (int): The size of negative sampling.
        cbow (bool): True for CBOW; False for Skip-gram.
        epochs (int): The number of iterations to train over.
        seed (int): The seed for the random number generator.
        workers (int): The number of worker threads to train the model.

    Returns:
        The trained Word2Vec model.
    """
    # Determine the training algorithm based on the cbow boolean
    # sg = 1 for Skip-gram, sg = 0 for CBOW
    sg = 0 if cbow else 1

    # Initialize and train the Word2Vec model using the full namespace
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
