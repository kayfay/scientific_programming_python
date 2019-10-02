"""
expsin_function(t)
expsin_a_function(t, a)
f2c_function(F)
c2f_function(C)
double(a)
test_sum_1k()
sum_1k(M)
roots_quadratic(a, b, c)
sum_function(args)
poly(_product_function(x, roots)
trapint1(f, a, b)
trapint2(f, a, b)
trapezint(f, a, b, n)
midpoint(f, a, b, n)
adaptive_trapezint(f, a, b, eps)
triangle_area(vertices)
pathlength(x, y)
heaviside_function(x)
"""
from __future__ import print_function


def expsin_function(t):
    from math import sin, pi, e
    return (e**-t)*sin(pi*t)


def expsin_a_function(t, a=1):
    from math import sin, pi, e
    return (e**(-t*a))*sin(pi*t)


def test_f2c_function():
    f = 32
    c = 0
    assert f2c_function(f) == 0
    assert c2f_function(c) == 32
    assert f2c_function(c2f_function(f)) == f
    assert c2f_function(f2c_function(c)) == c


def f2c_function(F):
    return 5/9*(F - 32)


def c2f_function(C):
    return 9/5*C + 32


def test_double():
    assert double(2) == 4
    assert abs(double(0.1) - 0.2) < 1E-15
    assert double([1, 2]) == [1, 2, 1, 2]
    assert double((1, 2)) == (1, 2, 1, 2)
    assert double(3+4j) == 6+8j
    assert double('hello') == 'hellohello'


def double(a):
    return a*2


def test_sum_1k():
    assert sum_1k(M=3) == 1/1 + 1/2 + 1/3


def sum_1k(M):
    tot = 0
    for k in range(1, M+1):
        tot += 1/k

    return tot


def test_roots():
    success_complex = roots_quadratic() == -1.0, -4.0
    msg_complex = "roots != -1.0, -4.0"
    success_nan = roots_quadratic(a=2, b=-6, c=5) == "nan", "nan"
    msg_nan = 'roots != "nan", "nan"'
    assert success_complex, msg_complex
    assert success_nan, msg_nan


def roots_quadratic(a=1, b=5, c=4):
    from numpy import sqrt

    # Take a quadratic function.
    def f(x):
        return a*x**2 + b*x + c

    sq_root = b**2.0 - 4.0*a*c # get sqrt to test for complex numbers

    if sq_root < 0:
        plus = (-b + sqrt(sq_root))/2.0*a
        minus = (-b - sqrt(sq_root))/2.0*a
        return (plus, minus)
    else:
        sq_root = eval(str(abs(sq_root))+"j")
        plus = (-b + sq_root)*0.5*a
        minus = (-b - sq_root)*0.5*a
        return (plus, minus)


def test_sum_function():
    assert sum_function([1, 3, 5, -5]) == 4
    assert sum_function([[1, 2], [4, 3], [8, 1]]) == [1, 2, 4, 3, 8, 1]
    assert sum_function(['Hello, ', 'World!']) == 'Hello, World!'


def sum_function(args):
    sum_total = type(args[0])()
    for arg in args:
        sum_total += arg
    print("{}".format(sum_total))
    return sum_total


def test_poly_product_function():
    x, roots = 1, [2, 4, 7]
    success = poly_product_function(x, roots) == -18
    msg = "x:{}, roots:{} in poly_product_function != -18".format(x, roots)
    assert success, msg


def poly_product_function(x, roots):
    poly = 1
    for r in roots:
        poly *= x - r
    return poly


def trapzint1(f, a=0, b=5):
    return (b-a)/2 * (f(a) + f(b))


def trapzint2(f, a=0, b=5):
    half = (a+b)/2
    area = lambda x, y: (b-a)/2 * (f(x) + f(y))
    area1 = area(a, half)
    area2 = area(half, b)
    return area1 + area2


def trapezint(f, a, b, n):
    h = (b - a)/float(n)
    xi = lambda i: a + i * h
    f_list = [f(xi(j)) + f(xi(j+1)) for j in range(1, n)]
    return sum((0.5*h) * z for z in f_list)


def test_trap():
    actual = trapezint(lambda x: x**2, 0, 5, 10)
    expected = 41.66668743750001
    eps = 1E-16
    assert abs(actual - expected) > eps


def midpointint(f, a, b, n):
    h = (b - a)/float(n)
    area_list = [f(a + i * h + 0.5 * h) for i in range(0, n)]
    area = h * sum(area_list)
    return area


def adaptive_trapezint(f, a, b, eps=1E-5):
    n_limit = 1000000 # Used to avoid infinite loop.
    n = 2
    integral_n = trapezint(f, a, b, n)
    integral_2n = trapeint(f, a, b, 2*n)
    diff = abs(integral_2n - integral_n)
    print("trapezoidal diff: {}".format(diff))

    while (diff > eps) and (n < n_limit):
        integral_n = trapeint(f, a, b, n)
        integral_2n = trapeint(f, a, b, n)
        diff = abs(integral_2n - integral_n)
        print("trapezoidal diff: {}".format(diff))
        n *= 2

    if diff <= eps:
        print("The integral computes to: {}".format(integral_2n))
        return n
    else:
        return -n # If not found returns negative n.


def triangle_area(vertices):
    x1, y1 = vertices[0][0], vertices[0][1]
    x2, y2 = vertices[1][0], vertices[1][1]
    x3, y3 = vertices[2][0], vertices[2][1]
    A = 0.5*abs(x2*y3-x3*y2-x1*y3+x3*y1+x1*y2-x2*y1)
    return A


def test_triangle_area():
    """
    Verity the area of a triangle with vertex coordinates
    (0, 0), (1,0), (0,2).
    """
    v1 = (0, 0); v2 = (1, 0); v3 = (0, 2)
    vertices = [v1, v2, v3]
    expected = 1
    computed = triangle_area(vertices)
    tol = 1E-14
    success = (expected - computed) < tol
    msg = 'computed area={:g} != {:g} (expected)'.format(computed, expected)
    assert success, msg


def pathlength(x, y):
    x_list = []
    for i, xi in enumerate(x):
        x_list.append(xi-x[i-1]**2)
    y_list = []
    for i, yi in enumerate(y):
        y_list.append(yi-y[i-1]**2)

    from math import sqrt
    return sum([sqrt(abs(i)) for i in list(map(sum, zip(x_list, y_list)))])


def test_pathlength():
    success = round(pathlength([1, 2], [2, 4])) == 5
    msg = "pathlength for [1, 2], [2, 4] != approx 5"
    assert success, msg


def heaviside_function(x):
    if x < 0:
        return 0
    else:
        return 1


def test_H():
    assert heaviside_function(-10) == 0
    assert heaviside_function(-10E-15) == 0
    assert heaviside_function(0) == 1
    assert heaviside_function(10E-15) == 1
    assert heaviside_function(10) == 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        print("g(0): {:g} g(1): {:g}" \
              .format(expsin_function(0), expsin_function(1)))

        print("g(0): {:g} g(1): {:g}" \
              .format(expsin_a_function(0, a=10), expsin_a_function(1, a=10)))

        roots_quadratic()
        sum_function([1, 1, 1])

    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        test_double()
        test_sum_1k()
        test_f2c_function()
        test_roots()
        test_sum_function()
        test_poly_product_function()
        test_trap()
        test_triangle_area()
        test_pathlength()

    else:
        pass
