#!/usr/bin/env python3
"""
Module for performing semantic search on a corpus of documents.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path: Path to the corpus of reference documents.
        sentence: Sentence from which to perform semantic search.

    Returns:
        The reference text of the document most similar to sentence.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    for filename in os.listdir(corpus_path):
        filepath = os.path.join(corpus_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    corpus_embeddings = model.encode(documents)
    query_embedding = model.encode(sentence)

    similarities = util.cos_sim(query_embedding, corpus_embeddings)
    best_match_idx = np.argmax(similarities[0].numpy())

    return documents[best_match_idx]
