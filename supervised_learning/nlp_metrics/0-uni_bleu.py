#!/usr/bin/env python3
"""Calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Args:
        references: list of reference translations
            each reference translation is a list of the words
            in the translation
        sentence: list containing the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sentence_len = len(sentence)

    # Count occurrences of each unique word in the sentence
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    # For each unique word, find the max count in any single reference
    max_ref_counts = {}
    for word in word_counts:
        max_count = 0
        for reference in references:
            ref_count = reference.count(word)
            if ref_count > max_count:
                max_count = ref_count
        max_ref_counts[word] = max_count

    # Clipped count: min of sentence count and max reference count
    clipped_count = sum(
        min(word_counts[word], max_ref_counts[word]) for word in word_counts
    )

    precision = clipped_count / sentence_len

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
