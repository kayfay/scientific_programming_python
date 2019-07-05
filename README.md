# Scientific Programming notebooks and python code
![Book Cover](https://hplgit.github.io/scipro-primer/figs/Primer5th_pic.jpg) 


Code and notes from reading 
[Hans Peter Langtangen's book](https://hplgit.github.io/scipro-primer/)
and book repo
[Software and material](https://github.com/hplgit/scipro-primer).

Contains an ipython notebook with some formulas to accompany calculations, functions and scripts for sections, and a test file.

`scipro-primer` is the source material github repo by Hans
`intro_materials` is a composit of topics introducing coding examples
`function_formulas` contains python functions of formulas

Examples of book contents as follows, also available as science_notebook.pdf

## Newton's Second Law of Motion

![$y(t) = v_0 t - \frac{1}{2} g t^2$](https://latex.codecogs.com/gif.latex?y%28t%29%20%3D%20v_0%20t%20-%20%5Cfrac%7B1%7D%7B2%7D%20g%20t%5E2)

```python
5 * 0.6 - 0.5 * 9.81 * 0.6 ** 2
```




    1.2342



## Height of an object

![$y(t)=v_0t-\frac{1}{2}gt^2$](https://latex.codecogs.com/gif.latex?y%28t%29%3Dv_0t-%5Cfrac%7B1%7D%7B2%7Dgt%5E2)

 * v0 as initial velocity of objects
 * g
acceleration of gravity
 * t as time


With y=0 as axis of object start when t=0
at initial time.

![$v_0t-\frac{1}{2}gt^2 = t(v_0-\frac{1}{2}gt)=0 \Rightarrow
t=0$](https://latex.codecogs.com/gif.latex?v_0t-%5Cfrac%7B1%7D%7B2%7Dgt%5E2%20%3D%20t%28v_0-%5Cfrac%7B1%7D%7B2%7Dgt%29%3D0%20%5CRightarrow%20t%3D0) or ![$t=\frac{v_0}{g}$](https://latex.codecogs.com/gif.latex?t%3D%5Cfrac%7Bv_0%7D%7Bg%7D)

 * time to move up and return to y=0, return seconds
is ![$\frac{2 v_0}{g}$](https://latex.codecogs.com/gif.latex?%5Cfrac%7B2%20v_0%7D%7Bg%7D)
 and restricted to ![$t \in \left[ 0, \  \frac{2v_{0}}{g}\right]$](https://latex.codecogs.com/gif.latex?t%20%5Cin%20%5Cleft%5B%200%2C%20%5C%20%5Cfrac%7B2v_%7B0%7D%7D%7Bg%7D%5Cright%5D)


```python
# variables for newton's second law of motion
v0 = 5
g = 9.81
t = 0.6
y = v0*t - 0.5*g*t**2
print(y)
```

    1.2342



```python
# or using good pythonic naming conventions
initial_velocity = 5
acceleration_of_gravity = 9.81
TIME = 0.6
VerticalPositionOfBall = initial_velocity*TIME - \
                         0.5*acceleration_of_gravity*TIME**2
print(VerticalPositionOfBall)
```

    1.2342


## Integral calculation
![$$\int_{-\infty}^1 e^{-x^2}dx{\thinspace .}$$](https://latex.codecogs.com/gif.latex?%5Cint_%7B-%5Cinfty%7D%5E1%20e%5E%7B-x%5E2%7Ddx%7B%5Cthinspace%20.%7D)


```python
from numpy import *

def integrate(f, a, b, n=100):
    """
    Integrate f from a to b
    using the Trapezoildal rule with n intervals.
    """
    x = linspace(a, b, n+1)  # coords of intervals
    h = x[1] - x[0]
    I = h*(sum(f(x)) - 0.5*(f(a) + f(b)))
    return I

# define integrand
def my_function(x):
    return exp(-x**2)

minus_infinity = -20 # aprox for minus infinity
I = integrate(my_function, minus_infinity, 1, n=1000)
print("value of integral:", I)
```

    value of integral: 1.6330240187288536



```python
# Celsius-Fahrenheit Conversion
C = 21
F = (9/5)*C + 32
print(F)
```

    69.80000000000001


## Time to reach height of ![$y_c$](https://latex.codecogs.com/gif.latex?y_c)

![$y_c =v_0 t - \frac{1}{2} g t^2$](https://latex.codecogs.com/gif.latex?y_c%20%3Dv_0%20t%20-%20%5Cfrac%7B1%7D%7B2%7D%20g%20t%5E2)

Quadratic
equation to solve.

![$\frac{1}{2}gt^2-v_0t+y_c=0$](https://latex.codecogs.com/gif.latex?y_c%20%3Dv_0%20t%20-%20%5Cfrac%7B1%7D%7B2%7D%20g%20t%5E2)

![$t_1=\Bigg(v_0-\sqrt{v_0^2-2gy_c}\Bigg)/g\quad$up$\quad(t=t_1)$](https://latex.codecogs.com/gif.latex?%24t_1%3D%5CBigg%28v_0-%5Csqrt%7Bv_0%5E2-2gy_c%7D%5CBigg%29/g%5Cquad%24up%24%5Cquad%28t%3Dt_1%29%24)

![https://latex.codecogs.com/gif.latex?%24t_2%3D%5CBigg%28v_0&plus;%5Csqrt%7Bv_0%5E2-2gy_c%7D%5CBigg%29/g%5Cquad%24down%24%5Cquad%28t%3Dt_2%3Et_1%29%24](https://latex.codecogs.com/gif.latex?%24t_2%3D%5CBigg%28v_0&plus;%5Csqrt%7Bv_0%5E2-2gy_c%7D%5CBigg%29/g%5Cquad%24down%24%5Cquad%28t%3Dt_2%3Et_1%29%24)

```python
v0 = 5
g = 9.81
yc = 0.2
import math
t1 = (v0 - math.sqrt(v0**2 - 2 * g * yc)) / g
t2 = (v0 + math.sqrt(v0**2 - 2 * g * yc)) / g
print('At t=%g s and %g s, the height is %g m.' % (t1, t2, yc))
```

    At t=0.0417064 s and 0.977662 s, the height is 0.2 m.


## The hyperbolic sine function ![$sinh(x) = \frac{1}{2}(e^x - e^{-x})$](https://latex.codecogs.com/gif.latex?sinh%28x%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%28e%5Ex%20-%20e%5E%7B-x%7D%29) and other math functions with right hand sides.


```python
from math import sinh, exp, e, pi
x = 2*pi
r1 = sinh(x)
r2 = 0.5*(exp(x) - exp(-x))
r3 = 0.5*(e**x - e**(-x))
print(r1, r2, r3) # with rounding errors
```

    267.74489404101644 267.74489404101644 267.7448940410163



```python
# Math functions for complex numbers
from scipy import *

from cmath import sqrt
sqrt(-1)  # complex number with cmath

from numpy.lib.scimath import sqrt
a = 1; b = 2; c = 100
r1 = (-b + sqrt(b**2 - 4*a*c))/(2*a)
r2 = (-b - sqrt(b**2 - 4*a*c))/(2*a)
print("""
t1={r1:g}
t2={r2:g}""".format(r1=r1, r2=r2))
```

    
    t1=-1+9.94987j
    t2=-1-9.94987j



```python
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
y = v0*t - Rational(1,2)*g*t**2
dydt = diff(y ,t)
print("At time", dydt)
print("acceleration:", diff(y,t,t)) # 2nd derivative
y2 = integrate(dydt, t)
print("integration of dydt wrt t", y2)

# convert to python function
v = lambdify([t, v0, g],  # arguments in v
             dydt)  # symbolic expression
print("As a function compute y = %g" % v(t=0, v0=5, g=9.81))
```

    At time -g*t + v0
    acceleration: -g
    integration of dydt wrt t -g*t**2/2 + t*v0
    As a function compute y = 5



```python
# equation solving for expression e=0, t unknown
from sympy import solve
roots = solve(y, t)  # e is y
print("""
If y = 0 for t then t solves y for [{},{}].

""".format(
            y.subs(t, roots[0]),
            y.subs(t, roots[1])
          ) )
```

    
    If y = 0 for t then t solves y for [0,0].
    
    


![$y(t)=v_0t-\frac{1}{2}gt^2$,$t \in [0, \frac{2 v_0}{g}]$](https://latex.codecogs.com/gif.latex?y%28t%29%3Dv_0t-%5Cfrac%7B1%7D%7B2%7Dgt%5E2%2C%20t%20%5Cin%20%5B0%2C%20%5Cfrac%7B2%20v_0%7D%7Bg%7D%5D)


```python
# Taylor series to the order n in a variable t around the point t0
from sympy import exp, sin, cos
f = exp(t)
f.series(t, 0, 3)
f_sin = exp(sin(t))
f_sin.series(t, 0, 8)
```




![$\displaystyle 1 + t + \frac{t^{2}}{2} - \frac{t^{4}}{8} - \frac{t^{5}}{15} - \frac{t^{6}}{240} + \frac{t^{7}}{90} + O\left(t^{8}\right)$](https://latex.codecogs.com/gif.latex?%5Cdisplaystyle%201%20&plus;%20t%20&plus;%20%5Cfrac%7Bt%5E%7B2%7D%7D%7B2%7D%20-%20%5Cfrac%7Bt%5E%7B4%7D%7D%7B8%7D%20-%20%5Cfrac%7Bt%5E%7B5%7D%7D%7B15%7D%20-%20%5Cfrac%7Bt%5E%7B6%7D%7D%7B240%7D%20&plus;%20%5Cfrac%7Bt%5E%7B7%7D%7D%7B90%7D%20&plus;%20O%5Cleft%28t%5E%7B8%7D%5Cright%29)



## Taylor Series Polynomial to approximate functions; 
![$1 + t + \frac{t^{2}}{2} -
\frac{t^{4}}{8} - \frac{t^{5}}{15} - \frac{t^{6}}{240} + \frac{t^{7}}{90} +
O\left(t^{8}\right)$](https://latex.codecogs.com/gif.latex?1%20&plus;%20t%20&plus;%20%5Cfrac%7Bt%5E%7B2%7D%7D%7B2%7D%20-%20%5Cfrac%7Bt%5E%7B4%7D%7D%7B8%7D%20-%20%5Cfrac%7Bt%5E%7B5%7D%7D%7B15%7D%20-%20%5Cfrac%7Bt%5E%7B6%7D%7D%7B240%7D%20&plus;%20%5Cfrac%7Bt%5E%7B7%7D%7D%7B90%7D%20&plus;%20O%5Cleft%28t%5E%7B8%7D%5Cright%29)


```python
# expanding and simplifying expressions
from sympy import simplify, expand
x, y = symbols('x y')
f = -sin(x) * sin(y) + cos(x) * cos(y)
print(f)
print(simplify(f))
print(expand(sin(x + y), trig=True))  # expand as trig funct
```

    -sin(x)*sin(y) + cos(x)*cos(y)
    cos(x + y)
    sin(x)*cos(y) + sin(y)*cos(x)


## Trajectory of an object
![$$f(x) = x tan \theta - \frac{1}{2 v^{2}_{0}}\cdot\frac{gx^2}{cos^{2}\theta} + y_0$$](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20x%20tan%20%5Ctheta%20-%20%5Cfrac%7B1%7D%7B2%20v%5E%7B2%7D_%7B0%7D%7D%20%5Ccdot%20%5Cfrac%7Bgx%5E2%7D%7Bcos%5E%7B2%7D%5Ctheta%7D%20&plus;%20y_0)


```python
# Trajectory of an object
g = 9.81      # m/s**2
v0 = 15       # km/h
theta = 60    # degree
x = 0.5       # m
y0 = 1        # m

print("""\
v0      = %.1f km/h
theta   = %d degree
y0      = %.1f m
x       = %.1f m\
""" % (v0, theta, y0, x))

from math import pi, tan, cos
v0 = v0/3.6             # km/h 1000/1 to m/s 1/60
theta = theta*pi/180    # degree to radians

y = x*tan(theta) - 1/(2*v0**2)*g*x**2/((cos(theta))**2)+y0
print("y       = %.1f m" % y)
```

    v0      = 15.0 km/h
    theta   = 60 degree
    y0      = 1.0 m
    x       = 0.5 m
    y       = 1.6 m


## Conversion from meters to British units


```python
# Convert meters to british length.
meters = 640
m = symbols('m')
in_m = m/(2.54)*100
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
```

    
    Given 640 meters conversions for;
    inches are 25196.85 in
    feet are 2099.74 ft
    yards are 699.91 yd
    miles are 0.398 m
    


## Gaussian function 
![$$f(x) = \frac{1}{\sqrt{2\pi}s} \text{exp} \Bigg[-{\frac{1}{2} \Big( \frac{x-m}{s} \Big)^2} \Bigg]$$](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7Ds%7D%20%5Ctext%7Bexp%7D%20%5CBigg%5B-%7B%5Cfrac%7B1%7D%7B2%7D%20%5CBig%28%20%5Cfrac%7Bx-m%7D%7Bs%7D%20%5CBig%29%5E2%7D%20%5CBigg%5D)


```python
from sympy import pi, exp, sqrt, symbols, lambdify

s, x, m = symbols("s x m")
y = 1/ (sqrt(2*pi)*s) * exp(-0.5*((x-m)/s)**2)
gaus_d = lambdify([m, s, x], y)
gaus_d(m = 0, s = 2, x = 1)
```




    0.1760326633821498



## Drag force due to air resistance on an object as the expression;

![$$F_d =\frac{1}{2} C_D \varrho A V^2$$](https://latex.codecogs.com/gif.latex?F_d%20%3D%5Cfrac%7B1%7D%7B2%7D%20C_D%20%5Cvarrho%20A%20V%5E2)

Where
 * ![$C_D$](https://latex.codecogs.com/gif.latex?C_D) drag coefficient (based on
roughness and shape)
   * As 0.4
 * ![$\varrho$](https://latex.codecogs.com/gif.latex?%5Cvarrho) is air density
   * Air density of
air is ![$\varrho$](https://latex.codecogs.com/gif.latex?\varrho) = 1.2 kg/![$m^{-3}$](https://latex.codecogs.com/gif.latex?m%5E%7B-3%7D)
 * V is velocity of the object 
 * A is the
cross-sectional area (normal to the velocity direction)
   * ![$A = \pi a^{2}$](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cpi%20a%5E%7B2%7D) for
an object with a radius $a$
   * ![$a$](https://latex.codecogs.com/gif.latex?%24a%24) = 11 cm

Gravity Force on an object with
mass ![$a$](https://latex.codecogs.com/gif.latex?%24m%24) is ![$F_g = mg$](https://latex.codecogs.com/gif.latex?F_g%20%3D%20mg)
Where
 * ![$g$](https://latex.codecogs.com/gif.latex?%24g%24) = 9.81 m/![s$^{-2}$](https://latex.codecogs.com/gif.latex?s%5E%7B-2%7D)
 * mass = 0.43kg

![$F_d$](https://latex.codecogs.com/gif.latex?F_d)
and ![$F_g$](https://latex.codecogs.com/gif.latex?F_g)
 results
in a difference relationship between air
resistance versus
gravity at impact
time

![$$\frac{kg}{m^{-3}} \qquad and \qquad \frac{m}{s^{-2}}$$](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bkg%7D%7Bm%5E%7B-3%7D%7D%20%5Cqquad%20and%20%5Cqquad%20%5Cfrac%7Bm%7D%7Bs%5E%7B-2%7D%7D)


```python
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

Fd_low_impact = drag_force(Cd=0.4,
                       Q=air_density,
                       A=x_area,
                       V=low_velocity)

Fd_high_impact = drag_force(Cd=0.4,
                       Q=air_density,
                       A=x_area,
                       V=high_velocity)

print("ratio of drag force=%.1f and gravity force=%.1f: %.1f" % \
      (Fd_low_impact, Fg, float(Fd_low_impact/Fg)))

print("ratio of drag force=%.1f and gravity force=%.1f: %.1f" % \
      (Fd_high_impact, Fg, float(Fd_high_impact/Fg)))
```

    ratio of drag force=6335.5 and gravity force=4.2: 1501.9
    ratio of drag force=101368.7 and gravity force=4.2: 24030.7


![$$t=\frac{M^{2/3} c \rho^{1/3}} {K \pi^2 (4\pi/3)^{2/3}}\log{\left[0.76 \frac{\left(T_o - T_w\right)}{- T_w + T_y} \right]}$$](https://latex.codecogs.com/gif.latex?t%3D%5Cfrac%7BM%5E%7B2/3%7D%20c%20%5Crho%5E%7B1/3%7D%7D%20%7BK%20%5Cpi%5E2%20%284%5Cpi/3%29%5E%7B2/3%7D%7D%5Clog%7B%5Cleft%5B0.76%20%5Cfrac%7B%5Cleft%28T_o%20-%20T_w%5Cright%29%7D%7B-%20T_w%20&plus;%20T_y%7D%20%5Cright%5D%7D)


```python
def critical_temp(init_temp=4, final_temp=70, water_temp=100,
                  mass=47, density=1.038, heat_capacity=3.7,
                  thermal_conductivity=5.4*10**-3):
    """
    Heating to a temperature with prevention to exceeding critical
    points. Be defining critial temperature points based on
    composition, e.g., 63 degrees celcius outter and 70 degrees
    celcius inner we can express temperature and time as a
    function.


    Calculates the time for the center critical temp as a function
        of temperature of applied heat where exceeding passes a critical point.

    t = (M**(2/3)*c*rho**(1/3)/(K*pi**2*(4*pi/3)**(2/3)))*(ln(0.76*((To-Tw)/(Ty-Tw))))

    Arguments:
        init_temp: initial temperature in C of object e.g., 4, 20
        final_temp: desired temperature in C of object e.g., 70
        water_temp: temp in C for boiling water as a conductive fluid e.g., 100
        mass: Mass in grams of an object, e.g., small: 47, large: 67
        density: rho in g cm**-3 of the object e.g., 1.038
        heat_capacity: c in J g**-1 K-1 e.g., 3.7
        thermal_conductivity: in W cm**-1 K**-1 e.g., 5.4*10**-3
    Returns: Time as a float in seconds to reach temperature Ty.
    """
    from sympy import symbols
    from sympy import lambdify
    from sympy import sympify
    from numpy import pi
    from math import log as ln # using ln to represent natural log

    # using non-pythonic math notation create variables
    M, c, rho, K, To, Tw, Ty = symbols("M c rho K To Tw Ty")
    # writing out the formula
    t = sympify('(M**(2/3)*c*rho**(1/3)/(K*pi**2*(4*pi/3)**(2/3)))*(ln(0.76*((To-Tw)/(Ty-Tw))))')
    # using symbolic formula representation to create a function
    time_for_Ty = lambdify([M, c, rho, K, To, Tw, Ty], t)
    # return the computed value
    return time_for_Ty(M=mass, c=heat_capacity, rho=density, K=thermal_conductivity,
                       To=init_temp, Tw=water_temp, Ty=final_temp)



```


```python
critical_temp()
```




    313.09454902221626




```python
critical_temp(init_temp=20)
```




    248.86253747844728




```python
critical_temp(mass=70)
```




    408.3278117759983




```python
critical_temp(init_temp=20, mass=70)
```




    324.55849416396666



## Newtons second law of motion in direction x and y, aka accelerations:

![$F_x = ma_x$](https://latex.codecogs.com/gif.latex?F_x%20%3D%20ma_x) is the sum of force, ```m*a_x (mass * acceleration)```

![$a_x = \frac {d^{2}x}{dt^{2}}$](https://latex.codecogs.com/gif.latex?a_x%20%3D%20%5Cfrac%20%7Bd%5E%7B2%7Dx%7D%7Bdt%5E%7B2%7D%7D), ```ax = (d**2*x)/(d*t**2)```

With gravity from ![$F_x$](https://latex.codecogs.com/gif.latex?F_x) as 0 as ![$x(t)$(https://latex.codecogs.com/gif.latex?x%28t%29) is in the horizontal position at time t

![$F_y = ma_y$](https://latex.codecogs.com/gif.latex?F_y%20%3D%20ma_y) is the sum of force, ```m*a_y```

![$a_y = \frac {d^{2}y}{dt^{2}}$](https://latex.codecogs.com/gif.latex?a_y%20%3D%20%5Cfrac%20%7Bd%5E%7B2%7Dy%7D%7Bdt%5E%7B2%7D%7D), ```ay = (d**2*y)/(d*t**2)```

With gravity from ![$F_y$](https://latex.codecogs.com/gif.latex?F_y) as ![$-mg$](https://latex.codecogs.com/gif.latex?-mg) since ![$y(t)$](https://latex.codecogs.com/gif.latex?y%28t%29) is in the veritcal postion at time t


Let coodinate ![$(x(t), y(t))$](https://latex.codecogs.com/gif.latex?%28x%28t%29%2C%20y%28t%29%29) be horizontal and verical positions to time t then we can integrate Newton's two components, ![$(x(t), t(t))$](https://latex.codecogs.com/gif.latex?%28x%28t%29%2C%20t%28t%29%29) using the second law twice with initial velocity and position with respect to t

![$\frac{d}{dt}x(0)=v_0 cos\theta$](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdt%7Dx%280%29%3Dv_0%20cos%5Ctheta)

![$\frac{d}{dt}y(0)=v_0 sin\theta$](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdt%7Dy%280%29%3Dv_0%20sin%5Ctheta)

![$x(0) = 0$](https://latex.codecogs.com/gif.latex?x%280%29%20%3D%200)

![$y(0) = y_0$](https://latex.codecogs.com/gif.latex?y%280%29%20%3D%20y_0)


    """
    Derive the trajectory of an object from basic physics.
        Newtons second law of motion in direction x and y, aka accelerations:
            F_x = ma_x is the sum of force, m*a_x (mass * acceleration)
            F_y = ma_y is the sum of force, m*a_y
        let coordinates (x(t), y(t)) be position horizontal and vertical to time t
        relations between acceleration, velocity, and position are derivatives of t

        $a_x = \frac {d^{2}x}{dt^{2}}$, ax = (d**2*x)/(d*t**2)
        $a_y = \frac {d^{2}y}{dt^{2}}$  ay = (d**2*y)/(d*t**2)

        With gravity and F_x = 0 and F_y = -mg

        integrate Newton's the two components, (x(t), y(t)) second law twice with
        initial velocity and position wrt t

        $\frac{d}{dt}x(0)=v_0 cos\theta$
        $\frac{d}{dt}y(0)=v_0 sin\theta$
        $x(0) = 0$
        $y(0) = y_0$

        Derivative(t)x(0) = v0*cos(theta) ; x(0) = 0
        Derivative(t)y(0) = v0*sin(theta) ; y(0) = y0

        from sympy import *
        diff(Symbol(v0)*cos(Symbol(theta)))
        diff(Symbol(v0)*sin(Symbol(theta)))

    theta: some angle, e.g, pi/2 or 90

    Return: relationship between x and y

    # the expression for x(t) and y(t)



    # if theta = pi/2 then motion is vertical e.g., the y position formula

    # if t = 0, or is eliminated then x and y are the object coordinates

    """

#### there isn't any code to this, it just looks at newtons second law of motion


## Sine function as a polynomial
![$$sin(x) \approx x - \frac{x^3}{3!} + \frac{x^5}{5!} + \frac{x^7}{7!} + \dotsb$$](https://latex.codecogs.com/gif.latex?sin%28x%29%20%5Capprox%20x%20-%20%5Cfrac%7Bx%5E3%7D%7B3%21%7D%20&plus;%20%5Cfrac%7Bx%5E5%7D%7B5%21%7D%20&plus;%20%5Cfrac%7Bx%5E7%7D%7B7%21%7D%20&plus;%20%5Cdotsb)


```python
x, N, k, sign = 1.2, 25, 1, 1.0
s = x
import math

while k < N:
    sign = - sign
    k = k + 2
    term = sign*x**x/math.factorial(k) 
    s = s + term

print("sin(%g) = %g (approximation with %d terms)" % (x, s, N))
```

    sin(1.2) = 1.0027 (approximation with 25 terms)


## Print table using an approximate Fahrenheit-Celcius conversiion.

For the approximate formula ![$C \approx \hat{C} = (F-30)/2$](https://latex.codecogs.com/gif.latex?C%20%5Capprox%20%5Chat%7BC%7D%20%3D%20%28F-30%29/2) farenheit to celcius conversions are calculated.  Adds a third to conversation_table with an approximate value ![$\hat{C}$](https://latex.codecogs.com/gif.latex?%5Chat%7BC%7D).


```python
F=0; step=10; end=100 # declare
print('------------------')
while F <= end:
    C = F/(9.0/5) - 32
    C_approx = (F-30)/2
    print("{:>3} {:>5.1f} {:>3.0f}".format(F, C, C_approx))
    F = F + step
print('------------------')
```

    ------------------
      0 -32.0 -15
     10 -26.4 -10
     20 -20.9  -5
     30 -15.3   0
     40  -9.8   5
     50  -4.2  10
     60   1.3  15
     70   6.9  20
     80  12.4  25
     90  18.0  30
    100  23.6  35
    ------------------


## Create sequences of odds from 1 to any number.


```python
n = 9 # specify any number
c = 1
while 1 <= n:
    if c%2 == 1:
        print(c)

    c += 1
    n -= 1
```

    1
    3
    5
    7
    9


## Compute energy levels in an atom
Compute the n-th energy level for an electron in an atom, e.g., 
Hydrogen: ![$$E_n = - \frac{m_{e}e^4}{8\epsilon_{0}^{2}h^2}\cdot\frac{1}{n^2}$$](https://latex.codecogs.com/gif.latex?E_n%20%3D%20-%20%5Cfrac%7Bm_%7Be%7De%5E4%7D%7B8%5Cepsilon_%7B0%7D%5E%7B2%7Dh%5E2%7D%5Ccdot%5Cfrac%7B1%7D%7Bn%5E2%7D)

where: 

![$m_e = 9.1094 \cdot 10^{-31}$](https://latex.codecogs.com/gif.latex?m_e%20%3D%209.1094%20%5Ccdot%2010%5E%7B-31%7D)kg is the electron mass

![$e = 1.6022 \cdot 10^{-19}$](https://latex.codecogs.com/gif.latex?e%20%3D%201.6022%20%5Ccdot%2010%5E%7B-19%7D)C is the elementary charge

![$\epsilon_0 = 8.8542 \cdot 10^{-12}$s$^2$](https://latex.codecogs.com/gif.latex?%5Cepsilon_0%20%3D%208.8542%20%5Ccdot%2010%5E%7B-12%7D%24s%24%5E2)![kg$^{-1}$m$^{-3}$](https://latex.codecogs.com/gif.latex?kg%5E%7B-1%7Dm%5E%7B-3%7D) is electrical permittivity of vacuum

![h = 6.6261 \cdot 10^{-34}$](https://latex.codecogs.com/gif.latex?h%20%3D%206.6261%20%5Ccdot%2010%5E%7B-34%7D)Js

Calculates energy level ![$E_n$](https://latex.codecogs.com/gif.latex?E_n) for ![$n= 1,...,20$](https://latex.codecogs.com/gif.latex?n%3D%201%2C...%2C20)


```python
def formula():
    # Symbolic computing
    from sympy import (
        symbols,  # define symbols for symbolic math
        lambdify,  # turn symbolic expr. into python functions
    )

    # declare symbolic variables
    m_e, e, epsilon_0, h, n = symbols('m_e e epsilon_0 h n')
    # formula
    En = -(m_e*e**4)/(8*epsilon_0*h**2)*(1/n**2)

    # convert to python function
    return lambdify([m_e, e, epsilon_0, h, n],  # arguments in En
                 En)  # symbolic expression

def compute_atomic_energy(m_e=9.094E-34, 
                          e=1.6022E-19, 
                          epsilon_0=9.9542E-12,
                          h=6.6261E-34):
    
    En = 0 # energy level of an atom
    for n in range(1, 20): # Compute for 1,...,20
        En += formula()(m_e, e, epsilon_0, h, n)
    
    return En
```

```python
compute_atomic_energy()
```




    -2.7315307541142e-32



and energy released moving from level ![$n_i$](https://latex.codecogs.com/gif.latex?n_i) to ![$n_f$](https://latex.codecogs.com/gif.latex?n_f) is 

![$$\Delta E = - \frac{m_{e}e^{4}}{8\epsilon_{0}^{2}h^{2}} \cdot \left( \frac{1}{n_{i}^{2}} - \frac{1}{n_{f}^{2}} \right)$$](https://latex.codecogs.com/gif.latex?%5CDelta%20E%20%3D%20-%20%5Cfrac%7Bm_%7Be%7De%5E%7B4%7D%7D%7B8%5Cepsilon_%7B0%7D%5E%7B2%7Dh%5E%7B2%7D%7D%20%5Ccdot%20%5Cleft%28%20%5Cfrac%7B1%7D%7Bn_%7Bi%7D%5E%7B2%7D%7D%20-%20%5Cfrac%7B1%7D%7Bn_%7Bf%7D%5E%7B2%7D%7D%20%5Cright%29)

```python
# Symbolic computing
from sympy import (
    symbols,  # define symbols for symbolic math
    lambdify,  # turn symbolic expr. into python functions
    )

# declare symbolic variables
m_e, e, epsilon_0, h, ni, nf = symbols('m_e e epsilon_0 h ni nf')
# formula
delta_E = -(m_e*e**4)/(8*epsilon_0*h**2)*((1/ni**2)-(1/nf**2))

# convert to python function
y = lambdify([m_e, e, epsilon_0, h, ni, nf],  # arguments in En
             delta_E)  # symbolic expression

def compute_change_in_energy(m_e=9.094E-34, 
                             e=1.6022E-19, 
                             epsilon_0=9.9542E-12,
                             h=6.6261E-34):
    print("Energy released going from level to level.")
    En = y(m_e, e, epsilon_0, h, 2, 1) # energy at level 1
    for n in range(2, 20): # Compute for 1,...,20
        En += y(m_e, e, epsilon_0, h, n-1, n)
        print("{:23.2E}  {:7} to level {:2}".format(
            y(m_e, e, epsilon_0, h, n-1, n),
            n-1,
            n))
    print("Total energy: {:.2E}".format(compute_atomic_energy()))
```

```python
compute_change_in_energy()
```

    Energy released going from level to level.
                  -1.29E-32        1 to level  2
                  -2.38E-33        2 to level  3
                  -8.33E-34        3 to level  4
                  -3.86E-34        4 to level  5
                  -2.09E-34        5 to level  6
                  -1.26E-34        6 to level  7
                  -8.20E-35        7 to level  8
                  -5.62E-35        8 to level  9
                  -4.02E-35        9 to level 10
                  -2.97E-35       10 to level 11
                  -2.26E-35       11 to level 12
                  -1.76E-35       12 to level 13
                  -1.40E-35       13 to level 14
                  -1.13E-35       14 to level 15
                  -9.22E-36       15 to level 16
                  -7.65E-36       16 to level 17
                  -6.41E-36       17 to level 18
                  -5.42E-36       18 to level 19
    Total energy: -2.73E-32
