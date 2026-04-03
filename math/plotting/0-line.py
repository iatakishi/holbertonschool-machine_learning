#!/usr/bin/env python3
"""line """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """ line """
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(x, y, 'r-')
    plt.show()
