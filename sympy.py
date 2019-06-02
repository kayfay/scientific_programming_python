# -*- coding: utf-8 -*-

# Symbolic computing
from sympy import (
    symbols,  # define symbols for symbolic math
    diff,  # differentiate expressions
    integrate,  # integrate expressions
    Rational,  # define rational numbers
    lambdify,  # turn symbolic expr. into python functions
)

# declare symbolic variables
t, v0, g = symbols('t v0 g')

# formula
y = v0 * t - Rational(1, 2) * g * t**2
dydt = diff(y, t)
print("at time", dydt)
print("acceleration:", diff(y, t, t))  # 2nd derivative
y2 = integrate(dydt, t)
print("integration of dydt wrt t", y2)


# convert to python function
v = lambdify(
    [t, v0, g],  # arguments in v
    dydt)  # symbolic expression
v(t=0, v0=5, g=9.81)

# equation solving for expression e=0, t unknown
from sympy import solve
roots = solve(y, t)  # e is y
y.subs(t, roots[0])
y.subs(t, roots[1])

# Taylor series to the order n in a variable t around the point t0
from sympy import exp, sin, cos
f = exp(t)
f.series(t, 0, 3)
f_sin = exp(sin(t))
f_sin.series(t, 0, 8)

# latex output per formula
from sympy import latex
print(latex(f_sin.series(t, 0, 8)))

# expanding and simplifying expressions
from sympy import simplify, expand
x, y = symbols('x y')
f = -sin(x) * sin(y) + cos(x) * cos(y)
print(simplify(f))
print(expand(sin(x + y), trig=True))  # expand as trig func
