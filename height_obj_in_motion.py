# -*- coding: utf-8 -*-
# Height of an object at a given time with math sqrt
import math as m

V0 = 5
G = 9.81
YC = 0.2
T1 = (V0 - m.sqrt(V0**2 - 2 * G * YC)) / G
T2 = (V0 + m.sqrt(V0**2 - 2 * G * YC)) / G

print('At t=%g s and %g s, the height is %g m.' % (T1, T2, YC))
