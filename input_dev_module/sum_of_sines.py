"""
To approximate a function by a sum of sines we consider a piecewise constant function;
f(t) = 1, 0 < t < T/2, 0, t = T/2, -1, T/2 < t < T

f(t) can be approximated by the sum;
S(t;n) = 4/pi * sigma_sum at i = 1 to n of 1/(2*i-1) * sin((2*(2*i-1)*pi*t)/T)

And it can be shown;
S(t;n) -> f(t) as n -> infinity


S(t, n, T) returns the value of S(t;n)
f(t, T) returns the value for computing f(t)

Program input takes arguments passed for n and alpha; n = 1, 3, 4, 10, 30, 100
alpha = 0.01, 0.25, 0.49 such that t = alpha*T with T = 2*pi.

Output returns a tabluar information showing error f(t) - S(t;n).
"""
