#!/usr/bin/env python3
"""Creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: one-hot numpy.ndarray of shape
        (m, classes) with correct labels
        logits: one-hot numpy.ndarray of shape
        (m, classes) with predicted labels

    Returns:
        confusion matrix of shape (classes, classes)
        where rows = correct labels,
        columns = predicted labels
    """
    return np.matmul(labels.T, logits)
