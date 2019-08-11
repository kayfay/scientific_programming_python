# -*- coding: utf-8 -*-
"""
Docstring: This is a testing area for the bisection algorithm.
"""

from __future__ import print_function

def bisection_function(f, a, b, eps):
    """Uses the bisection algorithm to find the root of a function.

    A function can be passed to the program and a set of values to
    check for roots. The bisection algorithm is used to find the
    root along a given range.

    Examples on usage:
    >>> def f(x):
    >>>    return 2*x - 3  # one root x=1.5
    >>> x, iter = bisection_function(f, a=0, b=10, eps=1E-5)

    >>> from bisection_function import bisection
    >>> x, iter = bisection(lamnda x: x**3 + 2*x -1, -10, -10, 1E-5)

    >>> python bisection.py "lambda x: x-(x-1)*sin(x)" -2 1 1E-5
    Found rootx=5.96046e-07 in 24 iterations

    Keyword arguments:
        arg f: is a function e.g., f(x) = lambda x: x - 1 - sin(x)
        arg a: is a start range to search for the root
        arg b: is a end range to search for the root
        arg eps: is an argument to search

    Returns tuple of root and iterations to find root:
        (m, i)

    """
    eps = 1E-6
    a, b = 0, 10

    fa = f(a)
    if fa*f(b) > 0:
        return None, 0

    i = 0
    while b-a > eps:
        i += 1
        m = (a + b)/2.0
        fm = f(m)
        if fa*fm <= 0:
            b = m
        else:
            a = m
            fa = fm
    x = m

    return m, i

def test_bisection():

    def f(x):
        """The function f(x) = x - 1 sin x"""
        from math import sin
        return 2*x - 3  # one root x=1.5
        # return x - 1 - sin(x) # root x=1.93456

    eps = 1E-5
    x_expected = 1.5
    x, iter = bisection_function(f, a=0, b=10, eps=eps)
    success = abs(x - x_expected) < eps # test within eps tolerance
    assert success, "found x={:g} != 1.5".format(x)

    if x is None:
        print("f(x) does not change sign in [{:g}, {:g}]".format(a, b))
    else:
        print("The root is ", x, "found in", iter, "iterations")
        print("f({:g})={:g})".format(x, f(x)))


def get_input():
    """Get f, a, b, eps from the command line."""
    try:
        f = eval(sys.argv[1])
        a = float(sys.argv[2])
        b = float(sys.argv[3])
        eps = float(sys.argv[4])
    except IndexError:
        print("Usage {:s} f a b eps".format(sys.argv[0]))
        sys.exit(1)
    return f, a, b, eps


if __name__ == "__main__":
    import sys
    # import math objects used in formulas for user defined
    from math import (acos, asin, atan, atan2, ceil, cos, \
                      cosh, exp, fabs, floor, log, log10, \
                      pi, pow, sin, sinh, sqrt, tan, tanh)

    if len(sys.argv) >= 2 and sys.argv[1] == "test":
        test_bisection()
    else:
        f, a, b, eps = get_input()
        x, iter = bisection_function(f, a, b, eps)
        print("Found rootx={:g} in {:d} iterations".format(x, iter))
