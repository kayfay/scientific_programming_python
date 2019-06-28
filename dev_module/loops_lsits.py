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
        print("{:.1f} {:.1f}".format(F, C))
        F = F + step
    print('------------------')


def _function():
    """Function Docstring

    Keyword arguments:

    Returns:
    """

    return None

if __name__ == "__main__":
    conversation_table()
    _function()
