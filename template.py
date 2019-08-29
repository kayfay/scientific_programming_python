# -*- coding: utf-8 -*-
"""
Docstring: This is a testing area.
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == 'verify':
        test_function()
    else:
        _function()
