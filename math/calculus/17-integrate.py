#!/usr/bin/env python3
""" integrate """


def poly_integral(poly, C=0):
    """ integrate """
    # Validate inputs
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None

    # Check all coefficients are numbers
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Start with [C] for the constant of integration
    integral = [float(C)] if C != 0 else [0]

    # Integrate each term: coeff / (power + 1)
    for power, coef in enumerate(poly):
        new_power = power + 1
        new_coef = coef / new_power

        # Extend integral list if needed
        while len(integral) <= new_power:
            integral.append(0)

        integral[new_power] = new_coef

    # Trim trailing zeros to make list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    # Convert floats that are whole numbers to int
    result = []
    for num in integral:
        if isinstance(num, float) and num.is_integer():
            result.append(int(num))
        else:
            result.append(num)

    return result
