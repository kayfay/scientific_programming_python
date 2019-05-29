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

symbols

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

# Trajectory of an object
g = 9.81  # m/s**2
v0 = 15  # km/h
theta = 60  # degree
x = 0.5  # m
y0 = 1  # m

print("""\
v0      = %.1f km/h
theta   = %d degree
y0      = %.1f m
x       = %.1f m\
""" % (v0, theta, y0, x))

from math import pi, tan, cos
v0 = v0 / 3.6  # km/h 1000/1 to m/s 1/60
theta = theta * pi / 180  # degree to radians

y = x * tan(theta) - 1 / (2 * v0**2) * g * x**2 / ((cos(theta))**2) + y0
print("y     = %.1f m" % y)

# Convert meters to british length.
meters = 640
m = symbols('m')
in_m = m / (2.54) * 100
ft_m = in_m / 12
yrd_m = ft_m / 3
bm_m = yrd_m / 1760

f_in_m = lambdify([m], in_m)
f_ft_m = lambdify([m], ft_m)
f_yrd_m = lambdify([m], yrd_m)
f_bm_m = lambdify([m], bm_m)

print("""
Given {meters:g} meters conversions for;
inches are {inches:.2f} in
feet are {feet:.2f} ft
yards are {yards:.2f} yd
miles are {miles:.3f} m
""".format(meters=meters,
           inches=f_in_m(meters),
           feet=f_ft_m(meters),
           yards=f_yrd_m(meters),
           miles=f_bm_m(meters)))

# Compute mass of some substance
# with density = mass / volume and density as g/cm**3
with open("scipro-primer/src/files/densities.dat", 'r') as content:
    densities = [line.strip() for line in content]
    elements = ["".join(e.split()[:-1]) for e in densities]
    masses = [float(m.split()[-1]) for m in densities]
    densities = dict(zip(elements, masses))

m = symbols('m')
y = m * 10
m_d = lambdify([m], y)
for element, mass in densities.items():
    print("1 liter of %s is %g" % (element, m_d(mass)))

# Growth of money in a bank
# A initial amount and p interest
A, p, n = symbols('A p n')
growth = lambdify([A, p, n], A * ((p / 100) + 1)**n)
growth(A=1000, p=0.05, n=3)

# polynomial factorization
from math import pi

h = 5.0  # height
b = 2.0  # base
r = 1.5  # radius

area_parallelogram = h * b
print("The area of the parallelogram is %.3f" % area_parallelogram)

area_square = b**2
print("The area of the square is %g" % area_square)

area_circle = pi * r**2
print("The area of the circle is %.3f" % area_circle)

volume_cone = 1.0 / 3 * pi * r**2 * h
print("The volume of the cone is %.3f" % volume_cone)

# Gaussian function
from sympy import (pi, exp, symbols, lambdify)

s, x, m = symbols("s x m")
y = 1 / (sqrt(2 * pi) * s) * exp(-0.5 * ((x - m) / s)**2)
gaus_d = lambdify([m, s, x], y)
gaus_d(m=0, s=2, x=1)

###############################################################
# Air resistance, Q density, V velocity, A cross-sectional area
# normal to velocity diretion, and drag coefficient (based on
# shape and roughness of surface)
###############################################################

from sympy import (Rational, lambdify, symbols, pi)

g = 9.81  # gravity in m/s**(-2)
air_density = 1.2  # kg/m**(-3)
a = 11  # radius in cm
x_area = pi * a**2  # cross-sectional area
m = 0.43  # mass in kg
Fg = m * g  # gravity force
high_velocity = 120 / 3.6  # impact velocity in km/h
low_velocity = 30 / 3.6  # impact velocity in km/h

Cd, Q, A, V = symbols("Cd Q A V")
y = Rational(1, 2) * Cd * Q * A * V**2
drag_force = lambdify([Cd, Q, A, V], y)

Fd_low_impact = drag_force(Cd=0.4, Q=air_density, A=x_area, V=low_velocity)

print("ratio of drag force=%.1f and gravity force=%.1f: %.1f" % \
      (Fd_low_impact, Fg, float(Fd_low_impact/Fg)))

Fd_high_impact = drag_force(Cd=0.4, Q=air_density, A=x_area, V=high_velocity)

print("ratio of drag force=%.1f and gravity force=%.1f: %.1f" % \
      (Fd_high_impact, Fg, float(Fd_high_impact/Fg)))

################################################################
# Heating to a temperature with prevention to exceeding critical
# points. Be defining critial temperature points based on
# composition, e.g., 63 degrees celcius outter and 70 degrees
# celcius innter we can express temperature and time as a
# function.
################################################################

 t = (M**Rational(2,3)*c*rho**Rational(1,3) / (K*pi**2 * (4*pi/3)**Rational(2,3) )) * (ln(0.76*((To-Tw) / (Ty-Tw))))

