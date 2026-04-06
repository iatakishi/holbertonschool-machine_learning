#!/usr/bin/env python3
""" all in one """
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():
    """ all in one """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.suptitle('All in One', fontsize='x-small')

    # Plot 1: y0 public
    plt.subplot(3, 2, 1)
    plt.plot(y0)
    plt.title('Plot 1 ', fontsize='x-small')
    plt.xlabel('Range', fontsize='x-small')
    plt.ylabel('y0', fontsize='x-small')

    # Plot 2: Scatter
    plt.subplot(3, 2, 2)
    plt.scatter(x1, y1, c='magenta')
    plt.title('Plot 2', fontsize='x-small')
    plt.xlabel('x1', fontsize='x-small')
    plt.ylabel('y1', fontsize='x-small')

    # Plot 3: Exponential Decay
    plt.subplot(3, 2, 3)
    plt.plot(x2, y2)
    plt.title('Plot 3', fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction Remaining', fontsize='x-small')
    plt.yscale('log')

    # Plot 4: Two decays
    plt.subplot(3, 2, 4)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g--', label='Ra-226')
    plt.title('Plot 4', fontsize='x-small')
    plt.xlabel('Time (years)', fontsize='x-small')
    plt.ylabel('Fraction remaining', fontsize='x-small')
    plt.legend(fontsize='x-small')
    plt.yscale('log')

    # Plot 5: Histogram (spans 2 columns)
    plt.subplot(3, 2, (5, 6))
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.title('Project A', fontsize='x-small')
    plt.xlabel('Grades', fontsize='x-small')
    plt.ylabel('Number of Students', fontsize='x-small')
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()
