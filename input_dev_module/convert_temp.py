# -*- coding: utf-8 -*-
"""
Six conversion functions between temperatures in Celsius, Kelvin,
and Fahrenheit: C2F, F2C, C2K, K2C, F2K, K2F

Usage: convert_temp.py Num Temperature

Examples
>>> python convert_temp.py 65 F
    18.0 C 291.0 K

>>> python convert_temp.py 293 K
    19.0 C 67.0 F

>>> python convert_temp.py 22 C
    71.0 F 295.0 K
"""
def C2F(C):
    """Celcius to Fahrenheit"""
    return (C * 9/5 + 32) // 1


def F2C(F):
    """Fahrenheit to Celcius"""
    return ((F -32) * 5/9) // 1


def C2K(C):
    """Celcius to Kelvin"""
    return (C + 273.15) // 1


def K2C(K):
    """Kelvin to Celcius"""
    return (K - 273.15) // 1


def F2K(F):
    """Fahrenheit to Kelvin"""
    return ((F-32) * 5/9 + 273.15) // 1


def K2F(K):
    """Kelvin to Fa)hrenheit"""
    return ((K - 273.15) * 9/5 + 32) // 1


def switch():
    """A helper function for displaying a UI"""
    def C_function():
        print(C2F(float(sys.argv[1])), "F", \
              C2K(float(sys.argv[1])), "K"),


    def F_function():
        print(F2C(float(sys.argv[1])), "C", \
              F2K(float(sys.argv[1])), "K")


    def K_function():
        print(K2C(float(sys.argv[1])), "C", \
              K2F(float(sys.argv[1])), "F")


    switcher = {
        "C": C_function,
        "F": F_function,
        "K": K_function
    }.get(sys.argv[2], "Specify format 73 F")

    switcher()


def test_conversion():
    tol = 1E-13
    f, c = 94, 22.7778
    assert C2F(F2C(f)) == f
    computed = K2C(C2K(c))
    expected = c
    assert abs(expected - computed) < tol
    computed = K2F(F2K(f))
    expected = f
    assert abs(expected - computed) < tol

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == 'verify':
        test_function()
    else:
        switch()
