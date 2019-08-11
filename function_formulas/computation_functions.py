# -*- coding: utf-8 -*-
"""
Calculation functions: 
	obj_traj() prints the trajectory of an object
	meters_to_brit_len() prints conversion of meters to british length
	compute_mass() prints computation the mass of some substances
	value_growth() prints computation the growth of interest rate
	common_areas() print computations of some common areas
	gaussian_function() computes the values for gaussian function
	air_resistance() prints the drag force and force gravity
        critical_temp() computes critical temperature from heating an object


"""
from __future__ import print_function

def obj_traj(gravity=9.81, initial_velocity=15, theta=60, h_coord_x=0.5, v_coord_y0=1):
    """
    Trajectory of an object
    Attributes: (passed as an integer or float)
        gravity in m/s**2
        initial_velocity in km/h
        theta in degrees
        horizontal coordinate, x in m
        initial_vertical coordinate, y0 in m
    Returns: None
    """

    print("""\
    v0      = %.1f km/h
    theta   = %d degree
    y0      = %.1f m
    x       = %.1f m\
    """ % (initial_velocity, theta, v_coord_y0, h_coord_x))

    from math import pi, tan, cos
    initial_velocity = initial_velocity / 3.6  # km/h 1000/1 to m/s 1/60
    theta = theta * pi / 180  # degree to radians

    y = h_coord_x * tan(theta) - 1 / (
        2 * initial_velocity**2) * gravity * h_coord_x**2 / (
            (cos(theta))**2) + v_coord_y0
    print("y     = %.1f m" % y)

def meters_to_brit_len(meters=640):
    """
    Convert meters to british length.
    Attributes: (passed as an integer or float)
        meters
    Returns: None
    """
    from sympy import symbols, lambdify
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

def compute_mass(data_file="scipro-primer/src-4th/files/densities.dat"):
    """
    Compute mass of some substance
    """
    from sympy import (symbols, lambdify)
    # with density = mass / volume and density as g/cm**3
    with open(data_file, 'r') as content:
        densities = [line.strip() for line in content]
        elements = ["".join(e.split()[:-1]) for e in densities]
        masses = [float(n.split()[-1]) for n in densities]
        densities = dict(zip(elements, masses))

    m = symbols('m')
    y = m * 10
    m_d = lambdify([m], y)
    for element, mass in densities.items():
        print("1 liter of %s is %g" % (element, m_d(mass)))

def value_growth(amount=1000, percent=0.05, num_yr=3):
    """
    Compute the growth of interest rate on an initial investment amount.
    Arguments
        amount: initial amount, default of 1000 dollars/units
        percent: percent per year interest rate , 0.05 in decimal format
        num_year: the number of years, e.g., 3 years of growth after investment
    """
    from sympy import (symbols, lambdify)
    # Growth of money in a bank
    # A initial amount and p interest
    A, p, n = symbols('A p n')
    growth = lambdify([A, p, n], A * ((p / 100) + 1)**n)
    growth(A=amount, p=percent, n=num_yr)

def common_areas(height=5.0, base=2.0, radius=1.5):
    """
    Prints the computations of some common areas.
    Arguments to pass with defaults as;
        height with 5.0 as an example height
        base with 2.0  as an example base
        radius with 1.5 as an example radius
    Returns: None
    """
    from math import pi

    area_parallelogram = height * base
    print("The area of the parallelogram is %.3f" % area_parallelogram)

    area_square = base**2
    print("The area of the square is %g" % area_square)

    area_circle = pi * radius**2
    print("The area of the circle is %.3f" % area_circle)

    volume_cone = 1.0 / 3 * pi * radius**2 * height
    print("The volume of the cone is %.3f" % volume_cone)

def gaussian_function(mean=0, var=2, obs=1):
    """
    The Guassian function or normal function plots values on a
    bell-shapped curve.
    Arugments;
       mean;  m as a real number e.g., 0
       var; s as a real number > 0  e.g., 2
       obs; x as an observed value e.g., 1
    Return a float for given parameters
    """
    from math import pi, sqrt
    from sympy import (exp, symbols, lambdify)

    s, x, m = symbols("s x m")
    y = 1 / (sqrt(2 * pi) * s) * exp(-0.5 * ((x - m) / s)**2)
    gaus_d = lambdify([m, s, x], y)
    return gaus_d(m=mean, s=var, x=obs)

def air_resistance(g=9.81, q=1.2, a=11, m=0.43, velocity=120):
    """
    Air resistance, Q density, V velocity, A cross-sectional area
    normal to velocity diretion, and drag coefficient (based on
    shape and roughness of surface)
    Arguments:
        g; gravity in m/s**(-2)
        q; as air density in kg/m**(-3)
        a; area radius in cm
        m; as mass in kg
        velocity; impact velocity in km/h
            which will be expressed in m/s conversion
            e.g., high velocity 120/3.6 or low velocity 30/3.6
    """
    from math import pi
    from sympy import (Rational, lambdify, symbols)

    x_area = pi * a**2  # cross-sectional area
    Fg = m * g  # force gravity
    high_velocity = float(velocity) / 3.6  # impact velocity in km/h
    low_velocity = float(velocity) / 3.6  # impact velocity in km/h

    Cd, Q, A, V = symbols("Cd Q A V")
    y = Rational(1, 2) * Cd * Q * A * V**2
    drag_force = lambdify([Cd, Q, A, V], y)

    Fd_low_impact = drag_force(Cd=0.4, Q=q, A=x_area, V=low_velocity)

    print("ratio of drag force=%.1f and force gravity.=%.1f: %.1f" % \
          (Fd_low_impact, Fg, float(Fd_low_impact/Fg)))

    Fd_high_impact = drag_force(Cd=0.4, Q=q, A=x_area, V=high_velocity)

    print("ratio of drag force=%.1f and force gravity=%.1f: %.1f" % \
          (Fd_high_impact, Fg, float(Fd_high_impact/Fg)))

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
    from numpy import log as ln # using ln

    # using non-pythonic math notation create variables
    M, c, rho, K, To, Tw, Ty = symbols("M c rho K To Tw Ty")
    # writing out the formula
    t = sympify('(M**(2/3)*c*rho**(1/3)/(K*pi**2*(4*pi/3)**(2/3)))*(ln(0.76*((To-Tw)/(Ty-Tw))))')
    # using symbolic formula representation to create a function
    time_for_Ty = lambdify([M, c, rho, K, To, Tw, Ty], t)
    # return the computed value
    return time_for_Ty(M=mass, c=heat_capacity, rho=density, K=thermal_conductivity,
                       To=init_temp, Tw=water_temp, Ty=final_temp)

