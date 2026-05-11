#!/usr/bin/env python3
""" f1 score """
import numpy as np


def f1_score(confusion):
    """ f1 score """
    # True Positives: diagonal elements
    tp = np.diag(confusion)

    # Total samples in the matrix
    total = confusion.sum()

    # Actual positives per class (row sums)
    actual_pos = confusion.sum(axis=1)

    # Predicted positives per class (column sums)
    predicted_pos = confusion.sum(axis=0)

    # False Positives: predicted as class i but actually something else
    fp = predicted_pos - tp

    # True Negatives: everything not in the actual positive row & not FP
    tn = total - actual_pos - fp

    precision = tp / predicted_pos

    recall = tp / actual_pos

    f1 = 2 * (precision + recall) / (precision + recall)

    return f1
