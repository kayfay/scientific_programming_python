# -*- coding: utf-8 -*-
"""This module uses loops and lists"""

from __future__ import print_function

def conversation_table(F=0, step=10, end=100):
    """Print table of farenheit-celcius conversions.

        Keyword arguments:
            F: Start value of temp.
            step: Increment of C values.
            end: End value of temp.

        Returns: None
    """

    print('------------------')
    while F <= end:
        C = F/(9.0/5) - 32
        print("{} {:.1f}".format(F, C))
        F = F + step
    print('------------------')


def approx_conv_table(F=0, step=10, end=100):
    """Print table using an approximate Fahrenheit-Celcius conversiion.

    For the approximate formula C is relative to approx_C = (F-30)/2
    farenheit to celcius conversions are calculated.  Adds a third to
    conversation_table with an approximate value approx_C.

        Keyword arguments:
            F: Start value of temp.
            step: Increment of C values.
            end: End value of temp.

        Returns: None
    """

    print('------------------')
    while F <= end:
        C = F/(9.0/5) - 32
        approx_C = (F-30)/2
        print("{} {:.1f} {:.0f}".format(F, C, approx_C))
        F = F + step
    print('------------------')

    return None

def lists_append():
    """Works with a list.

    Primes is a list with 2, 3, 5, 7, 11, 13 which loops, with p = 17
    assigned to the end of the list.

        Returns: None
    """

    primes = [2, 3, 5, 7, 11, 13]
    p = 17
    for prime in primes:
        print(prime)
    primes.append(p)
    for prime in primes:
        print(prime)

    return None

def odds(n=9):
    """Generate odd numbers from 1 to n.

    Odd numbers have a remainder of 1 when divided by two, so a counter
    is used to output based on using n as a reducer to a sequence.
    """

    c = 1
    while 1 <= n:
        if c%2 == 1:
            print(c)

        c += 1
        n -= 1

    return None

def formula():
    """Helper function for compute_atomic_energy

    Returns a formula to be called.
    """

    # Symbolic computing
    from sympy import (
        symbols,  # define symbols for symbolic math
        lambdify,  # turn symbolic expr. into python functions
    )

    # declare symbolic variables
    m_e, e, epsilon_0, h, n = symbols('m_e e epsilon_0 h n')
    # formula
    En = -(m_e*e**4)/(8*epsilon_0*h**2)*(1/n**2)

    # convert to python function
    return lambdify([m_e, e, epsilon_0, h, n],  # arguments in En
                    En)  # symbolic expression

def compute_atomic_energy(m_e=9.094E-34,
                          e=1.6022E-19,
                          epsilon_0=9.9542E-12,
                          h=6.6261E-34):
    """Compute energy levels in an atom

    Computes the n-th energy level for an electron in an atom, e.g.,
    Hydrogen: En = -(m_e*e**4)/(8*epsilon_0*h**2)*(1/n**2)
        where:
            m_e=9.094E-34 kg is the electron mass
            e=1.6022E-19C is the elementary charge
            epsilon_0=9.9542E-12 s^2 kg^{-1} m^{-3} is electrical permittivity of vacuum
            h=6.6261E-34Js heat/energy

        Calculates energy level En for n= 1,...,20
    """


    En = 0 # energy level of an atom
    for n in range(1, 20): # Compute for 1,...,20
        En += formula()(m_e, e, epsilon_0, h, n)

    return En

# Symbolic computing
from sympy import (
    symbols,  # define symbols for symbolic math
    lambdify,  # turn symbolic expr. into python functions
    )

# declare symbolic variables
m_e, e, epsilon_0, h, ni, nf = symbols('m_e e epsilon_0 h ni nf')
# formula
delta_E = -(m_e*e**4)/(8*epsilon_0*h**2)*((1/ni**2)-(1/nf**2))

# convert to python function
y = lambdify([m_e, e, epsilon_0, h, ni, nf],  # arguments in En
             delta_E)  # symbolic expression

def compute_change_in_energy(m_e=9.094E-34,
                             e=1.6022E-19,
                             epsilon_0=9.9542E-12,
                             h=6.6261E-34):
    """Creates a table with energy released by level and total energy.

    A display function for compute_atomic_energy.
    """

    print("Energy released going from level to level.")
    En = y(m_e, e, epsilon_0, h, 2, 1) # energy at level 1
    for n in range(2, 20): # Compute for 1,...,20
        En += y(m_e, e, epsilon_0, h, n-1, n)
        print("{:23.2E}  {:7} to level {:2}".format(
            y(m_e, e, epsilon_0, h, n-1, n),
            n-1,
            n))
    print("Total energy: {:.2E}".format(compute_atomic_energy()))


def _function():
    """Docstring Summary

    Doc String
    """


    return None



if __name__ == "__main__":
    _function()
