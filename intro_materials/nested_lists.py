# -*- coding: utf-8 -*-
"""
Nesting lists
"""
from __future__ import print_function
import pprint

# Globals
Cdegrees = range(-20, 41, 5)  # (-20, -25, ..., 35, 40)
Fdegrees = [(9.0/5)*C+ 32 for C in Cdegrees]

table = [Cdegrees, Fdegrees]

def construct_nested():
    """Construct nested list"""
    table = []
    for C, F in zip(Cdegrees, Fdegrees):
        table.append([C, F])
    pprint.pprint(table)

def nested_list_comprehension():
    """nested list comprehension"""
    table = [[C, F] for C, F in zip(Cdegrees, Fdegrees)]
    pprint.pprint(table)

def indexing_function():
    for C, F in table[Cdegrees.index(10):Cdegrees.index(35)]:
        print("%5d %5.1f".format(C, F))




if __name__ == "__main__":
    _function()
