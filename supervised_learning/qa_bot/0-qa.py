#!/usr/bin/env python3
"""
Module for question answering using BERT.
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a
    question.

    Args:
        question: A string containing the question to answer.
        reference: A string containing the reference document.

    Returns:
        A string containing the answer, or None if no answer is found.
    """
    model_url = 'https://tfhub.dev/see--/bert-uncased-tf2-qa/1'
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'

    model = hub.load(model_url)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    inputs = tokenizer(
        question, reference, return_tensors='tf'
    )

    outputs = model(
        inputs['input_ids'],
        inputs['attention_mask'],
        inputs['token_type_ids']
    )

    start_logits = outputs[0]
    end_logits = outputs[1]

    start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
    end_idx = tf.argmax(end_logits, axis=1).numpy()[0]

    if start_idx > end_idx:
        return None

    answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer
