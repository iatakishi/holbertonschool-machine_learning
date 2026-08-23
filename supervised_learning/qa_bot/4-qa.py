#!/usr/bin/env python3
"""
Module for answering questions from multiple reference texts.
"""

semantic_search = __import__('3-semantic_search').semantic_search
question_answer = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts.

    Args:
        corpus_path: The path to the corpus of reference documents.
    """
    while True:
        user_input = input("Q: ")

        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, user_input)
        answer = question_answer(user_input, reference)

        if answer is None or not answer.strip():
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
