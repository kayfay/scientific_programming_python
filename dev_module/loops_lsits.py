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

def _function():
    """Computes sums of first n integers.

    Computes the sum of integers from 1 to n, (inclusive), n(n + 1)/2
    """

    return None

if __name__ == "__main__":
    _function()
