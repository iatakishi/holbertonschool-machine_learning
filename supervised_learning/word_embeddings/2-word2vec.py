#!/usr/bin/env python3
"""Trains a Word2Vec model using gensim."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0,
                   workers=1):
    """
    Creates, builds and trains a gensim word2vec model.

    Parameters:
    sentences (list): list of sentences to be trained on
    vector_size (int): dimensionality of the embedding layer
    min_count (int): minimum number of occurrences of a word for
    use in training
    window (int): maximum distance between the current and
    predicted word within a sentence
    negative (int): size of negative sampling
    cbow (bool): True for CBOW, False for Skip-gram
    epochs (int): number of iterations to train over
    seed (int): seed for the random number generator
    workers (int): number of worker threads to train the model

    Returns:
    gensim.models.Word2Vec: the trained model
    """
    sg = 0 if cbow else 1

    if gensim.__version__.startswith('3.'):
        # Instantiate without sentences to prevent auto-training
        model = gensim.models.Word2Vec(
            size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            seed=seed,
            workers=workers,
            hashfxn=hash
        )
        # Explicitly build vocab and train to force deterministic behavior
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    else:
        # Gensim 4.x handles deterministic behavior out-of-the-box
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
