from sympy.physics.vector import dynamicsymbols
from sympy import diff, Symbol
from sympy.abc import t
from sympy import *

xt_1, xt_2, xt_3 = dynamicsymbols('x_1, x_2, x_3')
yt_1, yt_2, yt_3 = dynamicsymbols('y_1, y_2, y_3')

xt_1_dot = diff(xt_1, t)
xt_2_dot = diff(xt_2, t)
xt_3_dot = diff(xt_3, t)

yt_1_dot = diff(yt_1, t)
yt_2_dot = diff(yt_2, t)
yt_3_dot = diff(yt_3, t)

x_1, x_2, x_3 = symbols('x_1, x_2, x_3')
y_1, y_2, y_3 = symbols('y_1, y_2, y_3')

x_1_dot, x_2_dot, x_3_dot = symbols(r'\dot{x_1}, \dot{x_2}, \dot{x_3}')
y_1_dot, y_2_dot, y_3_dot = symbols(r'\dot{y_1}, \dot{y_2}, \dot{y_3}')

x_1_ddot, x_2_ddot, x_3_ddot = symbols(r'\ddot{x_1}, \ddot{x_2}, \ddot{x_3}')
y_1_ddot, y_2_ddot, y_3_ddot = symbols(r'\ddot{y_1}, \ddot{y_2}, \ddot{y_3}')

G_const = symbols('G', integer=True, positive=True)

m1_const = symbols('m_1', integer=True, positive=True)
m2_const = symbols('m_2', integer=True, positive=True)
m3_const = symbols('m_3', integer=True, positive=True)

r_square = ((xt_1 - xt_2)**2 + (yt_1 - yt_2)**2)

U = -G_const * (m1_const * m2_const)/ (r_square**(1/2))
T = (m1_const/2 * (xt_1_dot**2 + yt_1_dot**2) +\
     m2_const/2 * (xt_2_dot**2 + yt_2_dot**2))

q = [xt_1, yt_1, xt_2, yt_2]
q_dot = [xt_1_dot, yt_1_dot, xt_2_dot, yt_2_dot]

p = [x_1, y_1, x_2, y_2]
p_dot = [x_1_dot, y_1_dot, x_2_dot, y_2_dot]
p_ddot = [x_1_ddot,  y_1_ddot, x_2_ddot, y_2_ddot]




