# -*- coding: utf-8 -*-
"""
Docstring: This is a testing area.

 * plotting_parameters: pass parameters from command line to plot
"""
from __future__ import print_function

def _function():
    """This is a oneline docstring.

    This is a short summary of the function. These are some repeated lines.
    These are some repeated lines. These are some repeated lines. These
    are some repeated lines.

    Examples on usage:
    >>> def _function(arg1, arg2, arg3):
    >>>    return None, None, None
    >>> x, y, z = _function(a=0, b=10, c=20)

    Keyword arguments:
        arg 1: parameter definition
        arg 2: parameter definition
        arg 3: parameter definition

    Returns tuple None, None, None:
        (x, y, z)

    """
    return None


def test_function():
    """Some test case docstring"""
    success = _function() == "Some test case"
    msg = "{:s} failed".format(_function.__name__)
    assert success, msg


def plotting_parameters(v0, g=9.81):
    """Pass parameters from the command line and plot corresponding curves.

    From the command line, pass parameters to a plot for the function for
    the trajectory of an object; y(t) = v0t - 1/2gt**2 with t in [0, 2v0/g]
    with g = 9.81.

    Examples on usage:
    >>> test.py plot 10 9.81
    >>> [<matplotlib.lines.Line2D ... fig]

    Keyword arguments:
        v0: Velocity of the object
        g: Gravity of an object

    Returns a plot of the formula.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.linspace(0, 2*v0/g, 51)


    def y(t):
        return v0*t-0.5*g*t**2


    y = y(t)
    xmin = min(t)
    xmax = max(t)
    ymin = min(y)
    ymax = max(y)

    plt.axis(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.plot(t, y)


def write_table_to_file(f=lambda x, y: x+2*y, xmin=0, xmax=2, nx=4,
                        ymin=-1, ymax=2, ny=3, width=5, decimals=0,
                        filename='table.dat'):
    """Creates linearally spaced table saved to file.

    Given the left side and right side of a table, populates y decending and
    x ascending numbers from a function formula. x + 2y runs x from 0 to 2
    in steps of 0.5 and y from -1 to 2 in steps of 1.

        An example table is as follows;

                     2   4  4.5    5  5.5    6
                     1   2  2.5    3  3.5    4
                     0   0  0.5    1  1.5    2
                    -1  -2 -1.5   -1 -0.5    0

                         0  0.5    1  1.5    2


    Examples on usage:
    >>> write_table_to_file(f=labmda x, y: x+2*y,
                             xmin=0, xmax=2, nx=4,
                             ymin=-1, ymax=2, ny=3,
                             width=10, decimals=0, filename='table.dat')
    >>> File table.dat written to /path/dir/.

    Keyword arguments:
        f:        A declared function, which can be written as an anonymous
                   lambda function or declared using def and passed as a function.
        xmin:     The starting parameter on the bottom row x axis of the table.
        xmax:     The ending parametern on the bottom row x axis of the table.
        nx:       The number of evenly spaced points to generate for the bottom
                   row x axis of the table.
        ymin:     The starting parameter on the vertical column y axis of the table.
        ymax:     The ending parametern on the vertical column y axis of the table.
        ny:       The number of evenly spaced points to generate for the vertical
                   column y axis of the table.
        width:    The width between column digits.
        decimals: The number of decimal places for each digit.
        filename: The name of the file to be saved.
    """
    import numpy as np

    x = np.linspace(xmin, xmin, nx)
    y = np.linspace(ymin, ymin, ny)

    for col_element in y:
        print("{element:{width}.{decimals}g}".format(col_element, width=width, \
                                                    decimals=decimals), end="")
        for xi, yi in zip(x, y):
            print("{element:{width}.{decimals}g}".format(element=f(xi, yi), \
                                                         width=width, \
                                                         decimals=decimals), end="")
        print()
        if col_element == max(y):
            print()
            for row_element in x:
                print("{element:{width}.{decimals}g}".format(element=row_element, \
                                                            width=width, \
                                                            decimals=decimals), end="")

def test_write_table_to_flie():
    filename = 'tmp.data'
    write_table_to_file(f=lambda x, y: x+2*y,
                        xmin=0, xmax=2, nx=4,
                        ymin=-1, ymax=2, ny=3,
                        width=5, decimals=0,
                        filename='table.dat')
    with open(filename, 'r') as infile:
        computed = infile.read()
    expected = """\
2    4  4.5    5  5.5    6
1    2  2.5    3  3.5    4
0    0  0.5    1  1.5    2
1   -2 -1.5   -1 -0.5    0

     0  0.5    1  1.5    2

"""
    assert computed == expected



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == 'verify':
        test_function()
    elif len(sys.argv) > 2 and sys.argv[1] == 'plot':
        plotting_parameters(float(sys.argv[2]), float(sys.argv[3]))
    else:
        _function()
