#!/usr/bin/env python3
""" sensitivity """
import numpy as np


def sensitivity(confusion):
    """ sensitivity """
    return np.diag(confusion) / confusion.sum(axis=1)
