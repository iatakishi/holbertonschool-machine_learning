#!/usr/bin/env python3
"""Trains a Word2Vec model using gensim."""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0,
                   workers=1):
    """
    Creates, builds and trains a gensim word2vec model.
    """
    sg = 0 if cbow else 1

    # 1. CREATE: Modeli boş şəkildə yaradırıq (sentences parametrini vermirik!)
    try:
        # Gensim 3.x üçün (ALX/Holberton adətən bunu istifadə edir)
        model = gensim.models.Word2Vec(
            size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            seed=seed,
            workers=workers
        )
    except TypeError:
        # Gensim 4.x üçün
        model = gensim.models.Word2Vec(
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            seed=seed,
            workers=workers
        )

    # 2. BUILD: Lüğəti (vocab) ayrıca qururuq
    model.build_vocab(sentences)

    # 3. TRAIN: Modeli öyrədirik
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
