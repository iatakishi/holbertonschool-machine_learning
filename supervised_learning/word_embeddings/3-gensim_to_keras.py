#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer."""
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Parameters:
    model (gensim.models.Word2Vec): a trained gensim word2vec model

    Returns:
    keras.layers.Embedding: the trainable keras Embedding layer
    """
    vocab_size = len(model.wv)
    embedding_dim = model.wv.vector_size
    weights = [model.wv.vectors]

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=weights,
        trainable=True
    )

    return embedding_layer
