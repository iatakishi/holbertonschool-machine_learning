#!/usr/bin/env python3
"""
Module for a simple interactive question-answer loop.
"""

import sys


def chat_loop():
    """
    Runs the interactive question-answer loop.
    """
    while True:
        user_input = input("Q: ")

        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            sys.exit(0)

        print("A: ")


if __name__ == "__main__":
    chat_loop()