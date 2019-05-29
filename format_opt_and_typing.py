# -*- coding: utf-8 -*-

# Mutiple formatting options

# Height with newline formatting

V0 = 5
G = 9.81
T = 0.6
Y = V0 * T - 0.5 * G * T**2

# Types and Conversion
type(V0)
type(G)
round(G)
int(round(G))

# 2.X String formatting
print("""
At t=%0xz s, an object with
initial velocity v0=%d m/s
is located at the height %g m.
""" % (T, V0, Y))  # multi-line string

import math as m

YC = 0.2
T1 = (V0 - m.sqrt(V0**2 - 2 * G * YC)) / G
T2 = (V0 + m.sqrt(V0**2 - 2 * G * YC)) / G

print('At t= %.1e s and %.1E s, \nthe height is %g m.' % (T1, T2, YC))

# String format method

print("""
At t={t:.2f} s, an object with
initial velocity v0={v0:d} m/s
is located at the height {y:g} m.
""".format(t=T, v0=V0, y=Y))

print("At t={t:g} s, \nthe height of the object is {y:.2f} m".format(t=T, y=Y))

# Celsius-Fahrenheit Conversion

C = 21
F = (9 / 5) * C + 32

# Import Math Object
import math

m = math
ln = m.log
s = m.sin
c = m.cos

# Other math functions with right hand sides
# Hyperbolic sine function
# sinh(x) = 0.5*(m.e**x - m.e**-x)

from math import sinh, exp, e, pi

x = 2 * pi
r1 = sinh(x)
r2 = 0.5 * (exp(x) - exp(-x))
r3 = 0.5 * (e**x - e**(-x))

# value conversion
type("{x}".format(x=x))
type("{x!s}".format(x=x))
