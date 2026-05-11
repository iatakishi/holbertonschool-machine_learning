#!/usr/bin/env python3
""" f1 score """
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ f1 score """
    prec = precision(confusion)
    rec = sensitivity(confusion)
    return 2 * prec * rec / (prec + rec)
