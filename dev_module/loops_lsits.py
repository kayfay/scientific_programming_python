# -*- coding: utf-8 -*-
"""This module uses loops and lists"""

from __future__ import print_function

class ConvTable:
    def __init__(self, table_type, F, step, end):
        self.table_type = {'plain'  : plain,
                           'approx' : approx,
                           'typ_li' : typ_li}
        self.F = 0
        self.step = 10
        self.end = 100

    def plain(self, F, step, end):
        """Print table using Fahrenheit-Celcius conversiion.

           Keyword arguments:
                F: Start value of temp.
                step: Increment of C values.
                end: End value of temp.
        """

        print('------------------')
        while F <= end:
            C = F/(9.0/5) - 32
            print("{} {:.1f}".format(F, C))
            F = F + step
        print('------------------')

    def approx(self, F, end):
        """Print table using an approximate Fahrenheit-Celcius conversiion.

        For the approximate formula C is relative to approx_C = (F-30)/2
        farenheit to celcius conversions are calculated.  Adds a third to
        conversation_table with an approximate value approx_C.

            Keyword arguments:
                F: Start value of temp.
                step: Increment of C values.
                end: End value of temp.
        """

        print('------------------')
        while F <= end:
            C = F/(9.0/5) - 32
            approx_C = (F-30)/2
            print("{} {:.1f} {:.0f}".format(F, C, approx_C))
            F = F + step
        print('------------------')


    def typ_li(self, F, step, end):
        # Create lists for storing
        F, C, approx_C = [], [], []

        # Table
        print('------------------')
        while F <= end:
            # Use lists for storing
            C.append(F/(9.0/5) - 32)
            approx_C.append((F-30)/2)
            print("{} {:.1f} {:.0f}".format(F, C, approx_C))
            F = F + step
        print('------------------')

    def print_table(self, table_type='typ_li'):
        self.type.get(table_type, "nothing")()


def lists_append():
    """Works with a list.

    Primes is a list with 2, 3, 5, 7, 11, 13 which loops, with p = 17
    assigned to the end of the list.

        Returns: None
    """

    primes = [2, 3, 5, 7, 11, 13]
    p = 17
    for prime in primes:
        print(prime)
    primes.append(p)
    for prime in primes:
        print(prime)

    return None

def odds(n=9):
    """Generate odd numbers from 1 to n.

    Odd numbers have a remainder of 1 when divided by two, so a counter
    is used to output based on using n as a reducer to a sequence.
    """

    c = 1
    while 1 <= n:
        if c%2 == 1:
            print(c)

        c += 1
        n -= 1

    return None

def formula():
    """Helper function for compute_atomic_energy

    Returns a formula to be called.
    """

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
    """Compute energy levels in an atom

    Computes the n-th energy level for an electron in an atom, e.g.,
    Hydrogen: En = -(m_e*e**4)/(8*epsilon_0*h**2)*(1/n**2)
        where:
            m_e=9.094E-34 kg is the electron mass
            e=1.6022E-19C is the elementary charge
            epsilon_0=9.9542E-12 s^2 kg^{-1} m^{-3} is electrical permittivity of vacuum
            h=6.6261E-34Js heat/energy

        Calculates energy level En for n= 1,...,20
    """


    En = 0 # energy level of an atom
    for n in range(1, 20): # Compute for 1,...,20
        En += formula()(m_e, e, epsilon_0, h, n)

    return En

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
    """Creates a table with energy released by level and total energy.

    A display function for compute_atomic_energy.
    """

    print("Energy released going from level to level.")
    En = y(m_e, e, epsilon_0, h, 2, 1) # energy at level 1
    for n in range(2, 20): # Compute for 1,...,20
        En += y(m_e, e, epsilon_0, h, n-1, n)
        print("{:23.2E}  {:7} to level {:2}".format(
            y(m_e, e, epsilon_0, h, n-1, n),
            n-1,
            n))
    print("Total energy: {:.2E}".format(compute_atomic_energy()))


def linear_coor(a=0, b=5, n=10):
    """Generate equally spaced coordinates.

    Creates n + 1 equally spaced coordinates from [a, b], each interval has length
    h = (b - a)/n. the coordinates ar generated with xi = a + ih, i = 0,..., n + 1.

    Returns a list of floats or integers depending on division.
    """

    h = lambda b, a, n: (b - a)/n

    x = []
    for i in range(a, n + 1):
        x.append(a + i*h(b, a, n))

    return x


def obj_table1(g=9.81, v0=5, n=10, fr=True):
    """Makes a table from formula for the vector n object traveling in time.

    Prints a well formatted table for t and y(t) for the height of an object
        y(t) = v0*t-(1/2)*g*t**2.

    Argument g, v0, n are gravity, the null vector, and number of distance to span.
        fr implements True defaults to a for loop, False to a while loop.
    """

    # create values for [a, b] interval [0, 2*v0/g]
    a = 0
    b = 2*v0/g

    # object height
    print("  t  y(t)")
    print("*********")
    if fr:
        for i in range(n + 1):
            print("{:4.2f} {:4.2f}".format(i*(b - a)/n, # spaced time points t
                                           v0*(a + i*(b - a)/n)-(1/2)*g*(a + i*(b - a)/n)**2))

    else:
        count = 0
        while count < n + 1:
            print("{:4.2f} {:4.2f}".format(i*(b - a)/n, # spaced time points t
                                           v0*t-(1/2)*g*t**2))
            count += 1

def obj_table2(gravity=9.81, null_vector=5, num_positions=10, row_format=False):
    """Store values from a formula in lists.

    Stores t and y into two lists t and y, traverses lists with a for loop.
        Prints a well formatted table.

    Argument g, v0, n_positions are gravity, the null vector, and number of distance to span.
        row_format False defaults to return a column table format, a row table format.
            row_format and respectivly column format

    Returns: Two nested lists, a table representation of times and y heights.
    """

    from sympy import lambdify
    from sympy import symbols
    from sympy import Rational

    # variables for symbolic computation
    v0, ti, g = symbols("v0, ti, g")

    # create values for [a, b] interval [0, 2*v0/g]
    a = 0
    b = lambdify([v0, g], 2*v0/g)(null_vector, gravity) # calculate and return b

    # create formula for object height
    y_formula = v0*ti - Rational(1, 2)*g*ti**2

    # create a python function of the formula
    y_t = lambdify([v0, ti, g], y_formula)

    # uses linear_coor to create a list of times and list comp for y
    t = linear_coor(a=a, b=b, n=num_positions)
    y = [y_t(v0=null_vector, g=gravity, ti=time) for time in t]

    # table header
    if row_format:

        print("*"*92 + "\nt    :", end="")
        for col in t:
            # print t[1] some spaces t[2] ...
            print("{:4.2f}".format(col), end="\t")
            # print y[1] some spaces y[2] ...
        print("\ny(t) :", end="")
        for col in y:
            print("{:4.2f}".format(col), end="\t")
        print("\n" + "*"*92)

        return [[row[0], row[1]] for row in zip(t, y)]

    else:
        print("  t  y(t)")
        print("*********")

        for row in zip(t, y):
            print("{:4.2f} {:4.2f}".format(row[0], row[1]))
        print("*********")

        return [t, y]


def sim_comments():
    """Manual simulations of code blocks with comments."""

    a = [1, 3, 5, 7, 11] # list with 5 integers
    b = [13, 17] # list with 2 integers
    c = a + b # appends one list onto another
    print(c) # print out the appended list
    b[0] = -1 # set b[0], 13 to -1
    d = [e+1 for e in a] # add 1 each as a new list
    print(d) # [2, 4, 6, 8, 12]
    d.append(b[0] + 1) # add 0 to the end of d
    d.append(b[-1] + 1) # add 18 to the end of d
    print(d[-2:]) # print [-1, 17]
    for e1 in a: # e1 becomes [1,3,5,7,11] in seq
        for e2 in b: #e2 becomes [-1,17] in seq
            print(e1 + e2) # in sequence prints (1,-1),(1,17)...



def sigma_sum_while(k=1, M=100):
    """Replicates sigma sum in a while loop."""

    s = 0
    while k <= M:
        s += 1/k
        k += 1
    print(s)

def sigma_sum_for(k=1, M=100):
    """Replicates sigma sum in a while loop."""
    s = 0
    for k in range(1, M+1):
        s += 1/k
    print(s)

def interest_rate_loop():
    """Manual simulation of interest rate with a principal rate."""

    initial_amount = 100
    p = 5.5 # interest rate
    #  amount = initial_amount
    #  years = 0

    #  while amount <= 1.5*initial_amount:
    #      amount =+ p/100*amount
    #      years =+ 1

    # Alternate style loop
    value = [(0, 100)]
    while value[-1][1] <= 1.5 * initial_amount:
        value.append(
            (value[-1][0]+1,
             value[-1][1]+p/100*value[-1][1])
        ) # as index modifiers

    years = value[-1][0]
    print(years)

def inverse_sine():
    """Inverse sine function.

    The inverse ine or arcsine of x, sin**-1(x), since -1 <= sin(x) <= 1 for
    real x, the inverse sine is real-valued for -1 <= x <= 1, or defined
    monotonically increasing assuming values between -pi/2 and pi/2.
    """

    import mpmath
    print(mpmath.asin(-1), mpmath.asin(0), mpmath.asin(1))

def round_off_errors():
    # multiple sqrt and squaring results in rounding errors
    from math import sqrt
    for n in range(1, 60): # take 59 iterations of performing the same action
        r = 2.0
        for i in range(n): # use root 2
            r = sqrt(r)
        for i in range(n):
            r = r**2
        print("{:d} times sqrt and **2: {:.16f}".format(n, r))
    # 1 times sqrt and **2: 2.0000000000000004  <- rounding error
    # 59 times sqrt and **2: 1.0000000000000000 <- no longer the root 2

def zero():
    """Zero in computation epsilon

    The number with E-16 is as small as zero due to computation truncation.
       or machine epsilon, or machine zero.
    """

    eps = 1.0
    while 1.0 != 1.0 + eps: # use a while loop to iterate until computation truncation.
        print("............", eps)
        eps = eps/2.0 # reduce the amount by 2.0
    print("Final epsilon value:", eps) # Final epsilon value: 1.1102230246251565e-16

def real_num_tolerance():
    """Comparestwo real number tolerance"""
    a = 1/947.0*947
    b = 1
    if a != b:
        print("Wrong result!")

def interpret_time():
    """Uses the time module to time.


    """

    import time
    t0 = time.time()
    while time.time() - t0 < 10: # count 10 seconds since the time record was stored.
        print("....I like while loops!")
        time.sleep(2)
    print("Oh, no the - the loop stopped.")


if __name__ == "__main__":
    interpret_time()
