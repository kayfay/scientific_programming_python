# -*- coding: utf-8 -*-
"""
Various ways of handling input
"""
from __future__ import print_function

def f2c_qa_function():
    """Fahrenheit to Celsius conversion.

    Requests temperature in Fahrenheit degrees and computes temperature
    in Celsius degrees and prints in Celsius scale.

    Examples on usage:
    >>> f2c_qa_function()

    """
    F = float(input("Provide a Fahrenheit temperature in degrees: "))
    C = 5/9.0*F - 32
    print("The temperatire in Celcius is {:g}".format(C))


def f2c_cml_function():
    """Take an argument as input from the command line"""
    import sys

    F = float(sys.argv[1])
    C = 5/9.0*F - 32
    print("The temperatire in Celcius is {:g}".format(C))


def f2c_file_read_function():
    """Read temp from a file"""
    with open('data.txt', 'r') as infile:
        data = [i.strip().split() for i in infile] # store data as list

    F = float(data[-1][-1]) # last item in data should be value
    C = 5/9.0*F - 32
    print("The temperatire in Celcius is {:g}".format(C))


def f2c_file_read_write_function():
    """Read temp from a file"""
    with open('Fdeg.dat', 'r') as infile:
        data = [i.strip().split() for i in infile] # store data as list

    data = data[3:] # get lines with numerical values only

    F_list = [float(line[-1]) for line in data]
    C_list = [5/9.0*F - 32 for F in F_list]

    for i in range(len(C_list)):
        print("{:6g}F {:10.2f}C".format(F_list[i], C_list[i]))

    return F_list


def test_function():
    data = f2c_file_read_write_function()
    success = len(data) == 6
    assert success, "did not find 6 lines in data"


def f2c_cml_exc_function():
    """Take an argument as input from the command line"""
    import sys

    try:
        F = float(sys.argv[1])
        C = 5/9.0*F - 32
        print("The temperatire in Celcius is {:g}".format(C))
    except:
        print("Format should be {} with a temperature in Farenheit" \
              .format(sys.argv[0]))

def objects_qa_function():
    value = input("Provide input e.g., integer, real, " \
                  + "complex, list, tuple: ")
    print("Value is: {} Type is: {}".format(eval(value), type(eval(value))))


def objects_cml_function():
    value = eval(sys.argv[1])
    print("Value is: {} Type is: {}".format(value, type(value)))


def ball_qa_function():
    v0 = float(input("Provide an intial velocity: "))
    g = float(input("Provide a gravity: "))
    t = float(input("provide a time traveled: "))
    y = v0*t - 0.5*g*t**2
    print(y)


def ball_cml_function():
    v0 = float(sys.argv[1])
    g = 9.81
    t = float(sys.argv[2])
    y = v0*t - 0.5*g*t**2
    print(y)


def ball_cml_qa_function():
    try:
        v0 = float(sys.argv[1])
        t = float(sys.argv[2])

    except IndexError:
        v0 = float(input("Provide an intial velocity: "))
        t = float(input("provide a time traveled: "))

    g = 9.81
    y = v0*t - 0.5*g*t**2
    print(y)


def ball_cml_tcheck_function():
    v0 = float(sys.argv[1])
    g = 9.81
    t = float(sys.argv[2])

    if t >= 0 and t <= (2*v0/g):
        pass
    else:
        print("t does not lie between 0 and 2*v0/g")
        sys.exit()

    y = v0*t - 0.5*g*t**2
    print(y)


def ball_cml_ValueError_function():
    v0 = float(sys.argv[1])
    g = 9.81
    t = float(sys.argv[2])

    if t >= 0 and t <= (2*v0/g):
        pass
    else:
        raise ValueError("t does not lie between 0 and 2*v0/g")

    y = v0*t - 0.5*g*t**2
    print(y)


def ball_file_read_write_function():
    with open('ball.dat', 'r') as input_file:
        data = [line.strip().split() for line in input_file]

    v0 = float(data[0][1])
    t_list = [float(num) for sublist in data[3:] for num in sublist]

    return v0, t_list


def test_ball_file_read_write():
    v0, t_list = ball_file_read_write_function()
    with open('ball2.dat', 'w') as outfile:
        outfile.write('v0: {}\n'.format(v0))
        outfile.write('t:\n')
        for num in t_list:
            outfile.write('{} '.format(num))
        success = type(v0) == float and type(t_list) == list
        assert success, "type v0 is not a float or t_list is not a list"

def ball_file_read_write_function2():
    v0, t_list = ball_file_read_write_function()
    g = 9.81
    t_list.sort(reverse=False)
    with open('ball3.txt', 'w') as outfile:
        outfile.write("    t     |   y     \n")
        outfile.write("--------------------\n")
        for time in t_list:
            t = time
            y = v0*time - 0.5*g*time**2
            outfile.write('{:6.4f} {:10.4f}\n'.format(t, y))


def test_add():
    # Test integers
    assert add(1, 2) == 3

    # Test floating-point numbers with rounding errors
    tol = 1E-14
    a, b = 0.1, 0.2
    computed = add(a, b)
    expected = 0.3
    assert abs(expected - computed) < tol

    # Test lists
    assert add([1, 4], [4, 7]) == [1, 4, 4, 7]

    # Test strings
    assert add("Hello, ", "World") == "Hello, World"


def add(a, b):
    return a + b


def test_equal():
    assert equal('abc', 'abc') == (True, 'abc')
    assert equal('abc', 'aBc') == (False, 'ab|Bc')
    assert equal('abc', 'aBcd') == (False, 'ab|Bc*|d')
    assert equal('Hello, World!', 'hello world') == \
        (False, 'H|hello,|  |wW|oo|rr|ll|dd|*!|*')

def equal(a, b):
    # Find longest length and pad shorter string
    if len(a) > len(b):
        length = len(a)
        diff = length - len(b)
        b += diff*"*"
    elif len(a) < len(b):
        length = len(b)
        diff = length - len(a)
        a += diff*"*"
    else:
        length = len(a)

    # Set boolean value to return a true is equal
    isEqual = True

    return_string = "" # Use a string for building
    for index in range(0, length):

        # Compare until mismatch
        if a[index] == b[index]:
            return_string += a[index]
        else: # Append letter | letter
            return_string += a[index]
            return_string += "|"
            return_string += b[index]
            isEqual = False # Change if not true

    return isEqual, return_string


def test_stopping_length():
    assert stopping_length_function() == 188.77185034167707


def stopping_length_function(initial_velocity=120, friction_coefficient=0.3):
    """Newton's second law of motion for measuring stoppnig distance

    Newton's second law of motion is d = (1/2)*(v0**2/(mu*g)) so the stopping
    distance of an object in motion, like a car, can be measured.  The
    friction coefficient measures how slick a road is with a default of 0.3.

    Arguments:
        initial_velocity, v0: 120 km/h or 50 km/h
        friction_coefficient, mu: = 0.3

        >>> stopping_length(50, 0.3)
        188.77185034167707

        >>> stopping_length_function(50, 0.05)
        196.63734410591357


    Returns a real number as a floating point number.
    """
    g = 9.81
    v0 = initial_velocity/3.6
    mu = friction_coefficient

    return (1/2)*(v0**2/(mu*g))


def weekday_function():
    import calendar

    y = int(sys.argv[1])
    m = int(sys.argv[2])
    d = int(sys.argv[3])
    weekday = list(calendar.day_name)
    day_num = calendar.weekday(y, m, d)

    print(weekday[day_num])


def integrate_function():
    """Integration function

    Using scitools.StringFunction to do integration.

    >>> integration.py 'sin(x)' 0 pi/2
    integral of sin(x) on [0, 1.5708] with n=200: 1
    """
    def midpoint_integration(f, a, b, n=100):
        h = (b - a)/float(n)
        I = 0
        for i in range(n):
            I += f(a + i*h + 0.5*h)
        return h*I


    f_formula = sys.argv[1]
    a = eval(sys.argv[2])
    b = eval(sys.argv[3])
    if len (sys.argv) >= 5:
        n = int(sys.arvg[4])
    else:
        n = 200

    from scitools.StringFunction import StringFunction
    f = StringFunction(f_formula) # turn formula into f(x) func.

    """
    >>> g = StringFunction('A*exp(-a*t)*sin(omega*x)',
                       independent_variable='t',
                       A=1, a=0.1, omega=pi, x=0.5)
    >>> g.set_parameters(omega=0.1)
    >>> g.set_parameters(omega=0.1, A=5, x=0)
    >>> g(0)
    0.0
    >>> g(pi)
    2.8382392288852166e-15
    """

    I = midpoint_integration(f, a, b, n)
    print("Integral of {:s} on [{:g}, {:g}] with n ={:d}: {:g}" \
          .format(f_formula, a, b, n, I))


def unnamed_exception():
    try:
        C = float(sys.argv[1])
    except IndexError as error:
        print(error)
        print("C must be provided as command-line argument")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    import pprint
    from math import *

    if len(sys.argv) > 2 and sys.argv[1] == 'test':
        _function()
    else:
        unnamed_exception()
