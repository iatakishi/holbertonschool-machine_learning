#!/usr/bin/env python3
""" bars """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ bars """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    x = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]
    plt.figure(figsize=(6.4, 4.8))
    plt.bar(x, apples, label='apples', width=0.5, color='red')
    plt.bar(x, bananas, bottom=apples, label='bananas', width=0.5, color='yellow')
    plt.bar(x, oranges, bottom=apples+bananas, label='oranges', width=0.5, color='#ff8000')
    plt.bar(x, peaches, bottom=apples+bananas+oranges, label='peaches', width=0.5, color='#ffe5b4')
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend(loc='upper right')
    plt.show()
