#!/usr/bin/env python3
"""
Module for loading and tokenizing the Portuguese to English
translation dataset.
"""

import transformers
from setup import load_pt2en


class Dataset:
    """
    Class to load and prepare the dataset for machine translation.
    """

    def __init__(self):
        """
        Constructor for the Dataset class.
        Initializes training and validation datasets,
        and creates Portuguese and English tokenizers.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data: A tf.data.Dataset whose examples are formatted
                  as a tuple (pt, en).
                  pt is the tf.Tensor containing the Portuguese
                  sentence.
                  en is the tf.Tensor containing the corresponding
                  English sentence.

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
            pt_iterator(), vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
