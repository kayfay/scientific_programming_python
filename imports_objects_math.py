# -*- coding: utf-8 -*-
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

print(r1, r2, r3)
