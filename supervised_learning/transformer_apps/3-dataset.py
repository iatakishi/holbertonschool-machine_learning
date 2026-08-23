#!/usr/bin/env python3
"""
Module for loading, tokenizing, encoding, and pipelining the
Portuguese to English translation dataset.
"""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """
    Class to load and prepare the dataset for machine translation.
    """

    def __init__(self, batch_size, max_len):
        """
        Constructor for the Dataset class.
        Initializes datasets, creates tokenizers, encodes,
        filters, and batches the data.
        """
        raw_train = load_pt2en('train')
        raw_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(raw_train)

        train_encoded = raw_train.map(self.tf_encode)
        valid_encoded = raw_valid.map(self.tf_encode)

        def filter_fn(pt, en):
            return tf.logical_and(
                tf.shape(pt)[0] <= max_len,
                tf.shape(en)[0] <= max_len
            )

        train_filtered = train_encoded.filter(filter_fn)
        valid_filtered = valid_encoded.filter(filter_fn)

        self.data_train = train_filtered.cache() \
            .shuffle(20000) \
            .padded_batch(batch_size) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = valid_filtered \
            .padded_batch(batch_size)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data: A tf.data.Dataset whose examples are formatted
                  as a tuple (pt, en).

        Returns:
            tokenizer_pt: The Portuguese tokenizer.
            tokenizer_en: The English tokenizer.
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            for pt, en in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            for pt, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_tokens: list containing the Portuguese tokens.
            en_tokens: list containing the English tokens.
        """
        vocab_size = self.tokenizer_pt.vocab_size

        pt_str = pt.numpy().decode('utf-8')
        en_str = en.numpy().decode('utf-8')

        pt_encoded = self.tokenizer_pt.encode(
            pt_str, add_special_tokens=False
        )
        en_encoded = self.tokenizer_en.encode(
            en_str, add_special_tokens=False
        )

        pt_tokens = [vocab_size] + pt_encoded + [vocab_size + 1]
        en_tokens = [vocab_size] + en_encoded + [vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the corresponding English sentence.

        Returns:
            pt_encoded: tf.Tensor of Portuguese tokens.
            en_encoded: tf.Tensor of English tokens.
        """
        pt_encoded, en_encoded = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
