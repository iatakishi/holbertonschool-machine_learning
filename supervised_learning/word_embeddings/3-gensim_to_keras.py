#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer.

    Parameters:
    model (gensim.models.Word2Vec): a trained gensim word2vec model

    Returns:
    keras.layers.Embedding: the trainable keras Embedding layer
    """
    vocab_size = model.wv.vectors.shape[0]
    embedding_dim = model.wv.vector_size
    weights = [model.wv.vectors]

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=weights,
        trainable=True
    )

    return embedding_layer
