# -*- coding: utf-8 -*-
"""Newton's second law of motion:
y(t) = v0*t - (1/2)*g*t**2
 * v0 as initial velocity of objects
 * g acceleration of gravity
 * t as time

With y=0 as axis of object start when t=0 at initial time.

As a transform for a function of motion and return to initial state:
v0*t - (1/2)*g*t**2 = t(v0 - (1/2)*g*t) = 0 => t=0 or t = 2*(v0/g)

 * time to move up and return to y=0, return seconds is 2*(v0/g)
 and restricted to t exists in [0, 2*(v0/g)]
"""
# Variables for newton's second law of motion,
# computes height of an object in vertical motion.

V0 = 5  # initial velocity
G = 9.81  # acceleration of gravity
T = 0.6  # time
Y = V0 * T - 0.5 * G * T**2  # vertical position
print('At t=%g s, the height of the object is %.2f m.' % (T, Y))

# Using good pythonic naming conventions.
initial_velocity = 5
acceleration_of_gravity = 9.81
TIME = 0.6  # As a constant
VerticalPositionOfBall = initial_velocity*TIME - \
                         0.5*acceleration_of_gravity*TIME**2
print(VerticalPositionOfBall)
