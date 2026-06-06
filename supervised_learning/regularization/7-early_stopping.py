#!/usr/bin/env python3
"""Module for early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Arguments:
    cost -- current validation cost of the neural network
    opt_cost -- lowest recorded validation cost of the neural network
    threshold -- threshold used for early stopping
    patience -- patience count used for early stopping
    count -- count of how long the threshold has not been met

    Returns:
    A boolean of whether the network should be stopped early,
    followed by the updated count
    """
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return True, count

    return False, count
