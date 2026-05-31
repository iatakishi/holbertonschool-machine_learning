#!/usr/bin/env python3
"""Module for learning rate decay"""
import numpy as np


# Update learning rate using inverse time decay
def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Update the learning rate using stepwise inverse time decay"""
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
