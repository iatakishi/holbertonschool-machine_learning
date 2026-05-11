#!/usr/bin/env python3
""" specificity """
import numpy as np


def specificity(confusion):
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

    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)
