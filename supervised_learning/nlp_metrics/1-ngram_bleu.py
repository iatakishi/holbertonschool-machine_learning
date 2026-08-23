#!/usr/bin/env python3
"""Calculates the n-gram BLEU score for a sentence"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Args:
        references: list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns:
        the n-gram BLEU score
    """
    sentence_len = len(sentence)

    def get_ngrams(words, n):
        """Builds a list of n-grams (as tuples) from a list of words"""
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    sentence_ngrams = get_ngrams(sentence, n)
    reference_ngrams = [get_ngrams(reference, n) for reference in references]

    # Count occurrences of each unique n-gram in the sentence
    ngram_counts = {}
    for ngram in sentence_ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    # For each unique n-gram, find the max count in any single reference
    max_ref_counts = {}
    for ngram in ngram_counts:
        max_count = 0
        for ref_ngrams in reference_ngrams:
            ref_count = ref_ngrams.count(ngram)
            if ref_count > max_count:
                max_count = ref_count
        max_ref_counts[ngram] = max_count

    # Clipped count: min of sentence count and max reference count
    clipped_count = sum(
        min(ngram_counts[ngram], max_ref_counts[ngram])
        for ngram in ngram_counts
    )

    total_ngrams = len(sentence_ngrams)
    precision = clipped_count / total_ngrams

    # Find reference length closest to the sentence length
    ref_lens = [len(reference) for reference in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
    )

    # Brevity penalty
    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (closest_ref_len / sentence_len))

    bleu = bp * precision

    return bleu
