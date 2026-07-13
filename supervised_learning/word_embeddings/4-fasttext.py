#!/usr/bin/env python3
"""Trains a FastText model using gensim."""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model.
    """
    sg = 0 if cbow else 1

    # 1. CREATE: Modeli boş şəkildə yaradırıq
    try:
        # Gensim 3.x versiyası üçün (çox vaxt yoxlayıcı sistemlər bunu istifadə edir)
        model = gensim.models.FastText(
            size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            seed=seed,
            workers=workers
        )
    except TypeError:
        # Gensim 4.x versiyası üçün
        model = gensim.models.FastText(
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=sg,
            seed=seed,
            workers=workers
        )

    # 2. BUILD: Lüğəti müstəqil şəkildə qururuq
    model.build_vocab(sentences)

    # 3. TRAIN: Modeli öyrədirik (ikiqat öyrədilmənin qarşısını alırıq)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
