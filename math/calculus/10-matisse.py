#!/usr/bin/env python3
""" matisse """


def poly_derivative(poly):
    """ calculates the derivative of a polynomial """
    # 1. Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # 2. Calculate derivative
    new_power = []
    for i in range(1, len(poly)):
        new_power.append(i * poly[i])

    # 3. If the result is empty, return [0]
    if not new_power:
        return [0]

    return new_power
