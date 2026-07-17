#!/usr/bin/env python3
"""Calculates the cumulative n-gram BLEU score for a sentence"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references: list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    sentence_len = len(sentence)

    def get_ngrams(words, size):
        """Builds a list of n-grams (as tuples) from a list of words"""
        return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]

    def ngram_precision(size):
        """Computes the clipped n-gram precision for a given n-gram size"""
        sentence_ngrams = get_ngrams(sentence, size)
        reference_ngrams = [get_ngrams(reference, size)
                             for reference in references]

        ngram_counts = {}
        for ngram in sentence_ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        max_ref_counts = {}
        for ngram in ngram_counts:
            max_count = 0
            for ref_ngrams in reference_ngrams:
                ref_count = ref_ngrams.count(ngram)
                if ref_count > max_count:
                    max_count = ref_count
            max_ref_counts[ngram] = max_count

        clipped_count = sum(
            min(ngram_counts[ngram], max_ref_counts[ngram])
            for ngram in ngram_counts
        )

        total_ngrams = len(sentence_ngrams)
        return clipped_count / total_ngrams

    # Precision for each n-gram size from 1 to n, weighted evenly
    precisions = [ngram_precision(size) for size in range(1, n + 1)]

    # Geometric mean of the precisions
    geo_mean = np.exp(np.sum(np.log(precisions)) / n)

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

    bleu = bp * geo_mean

    return bleu
