#!/usr/bin/env python3
""" precision """
import numpy as np


def precision(confusion):
    """ confusion """
    return np.diag(confusion) / confusion.sum(axis=0)
