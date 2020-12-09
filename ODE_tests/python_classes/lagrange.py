import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy import linalg as LA

from mpmath import *

import sympy as sym
from sympy import *
from sympy.abc import t
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy import diff, Symbol
from sympy.printing.mathml import print_mathml
from sympy.physics.vector import dynamicsymbols
from sympy import Function

from IPython.display import display, Math

class Lagrange_finder():
    def __init__(self, q, q_dot, p, p_dot, p_ddot):
        self.q = q
        self.q_dot = q_dot
        self.p = p
        self.p_dot = p_dot
        self.p_ddot = p_ddot
    
    def Lagrangian(self, U, T, debug=False):
        self.Lagrangian = simplify(T - U)
        
        if(debug):
            return self.Lagrangian

    def Lagrange(self):
        self.ODE=[]

        for i in range(0, len(self.q)):
            dL_dq = diff(self.Lagrangian, self.q[i])
            dL_dq_dot = simplify(diff(self.Lagrangian, self.q_dot[i]))
            dL_dq_dot_dt = diff(dL_dq_dot, t)
            lagrange = simplify(Eq(dL_dq,dL_dq_dot_dt))

            for j in range(len(self.q)):
                lagrange = lagrange.subs([(diff(self.q_dot[j],t), self.p_ddot[j]),
                                          (self.q_dot[j], self.p_dot[j]),
                                          (self.q[j],self.p[j])])

            diffeq = solve(lagrange, self.p_ddot[i])
            self.ODE.append(diffeq)

        return self.ODE 

class two_body(Lagrange_finder):
    def __init__(self, q, q_dot, p, p_dot, p_ddot, G, m_1, m_2):
        Lagrange_finder.__init__(self, q = q, q_dot = q_dot,
                                 p = p, p_dot = p_dot, p_ddot = p_ddot)

        self.p = p
        self.p_dot = p_dot
        self.p_ddot = p_ddot

        self.G = G
        self.m_1 = m_1
        self.m_2 = m_2

    def conversion(self):
        numODE = []
        for i in range(len(self.ODE)):
            numODE.append(self.ODE[i][0])

        self.npfunc = []

        for i in range(len(numODE)):
            f = lambdify((self.p[0], self.p[1], self.p[2], self.p[3],
                         'G', 'm_1', 'm_2') , numODE[i], np)
            self.npfunc.append(f)

        return self.npfunc

    def dU_dt(self, t, U):
        acceleration = self.npfunc

        return [U[1], acceleration[0](U[0], U[2], U[4], U[6],
                                      self.G, self.m_1, self.m_2),
                U[3], acceleration[1](U[0], U[2], U[4], U[6],
                                      self.G, self.m_1, self.m_2),
                U[5], acceleration[2](U[0], U[2], U[4], U[6],
                                      self.G, self.m_1, self.m_2),
                U[7], acceleration[3](U[0], U[2], U[4], U[6],
                                      self.G, self.m_1, self.m_2)]
    def solve_ode_Leapfrog(self, t_pts,
                           x1_0, x1_0_dot, y1_0, y1_0_dot,
                           x2_0, x2_0_dot, y2_0, y2_0_dot):

        delta_t = t_pts[1] - t_pts[0]
        num_t_pts = len(t_pts)

        #body one
        x1 = np.zeros(num_t_pts)
        x1_dot = np.zeros(num_t_pts)
        x1_dot_half = np.zeros(num_t_pts)
        y1 = np.zeros(num_t_pts)
        y1_dot = np.zeros(num_t_pts)
        y1_dot_half = np.zeros(num_t_pts)

        #body two
        x2 = np.zeros(num_t_pts)
        x2_dot = np.zeros(num_t_pts)
        x2_dot_half = np.zeros(num_t_pts)
        y2 = np.zeros(num_t_pts)
        y2_dot = np.zeros(num_t_pts)
        y2_dot_half = np.zeros(num_t_pts)

        ### initial conditions ###
        # Body one
        x1[0] = x1_0
        x1_dot[0] = x1_0_dot
        y1[0] = y1_0
        y1_dot[0] = y1_0_dot

        #Body two
        x2[0] = x2_0
        x2_dot[0] = x2_0_dot
        y2[0] = y2_0
        y2_dot[0] = y2_0_dot

        #step through differential equation
        for i in np.arange(num_t_pts - 1):
            # time point
            t = t_pts

            U = [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]

            #Body one
            x1_dot_half[i] = x1_dot[i] + self.dU_dt(t, U)[1] * delta_t/2.
            x1[i+1] = x1[i] + x1_dot_half[i] * delta_t
            y1_dot_half[i] = y1_dot[i] + self.dU_dt(t, U)[3] * delta_t/2.
            y1[i+1] = y1[i] + y1_dot_half[i] * delta_t

            #Body two
            x2_dot_half[i] = x2_dot[i] + self.dU_dt(t, U)[5] * delta_t/2.
            x2[i+1] = x2[i] + x2_dot_half[i] * delta_t
            y2_dot_half[i] = y2_dot[i] + self.dU_dt(t, U)[7] * delta_t/2.
            y2[i+1] = y2[i] + y2_dot_half[i] * delta_t

            #update vector U
            U = [x1[i+1], x1_dot[i], y1[i+1], y1_dot[i],
                 x2[i+1], x2_dot[i], y2[i+1], y2_dot[i]]

            #update velocities
            #Body one
            x1_dot[i+1] = x1_dot_half[i] + self.dU_dt(t, U)[1] * delta_t/2.
            y1_dot[i+1] = y1_dot_half[i] + self.dU_dt(t, U)[3] * delta_t/2.

            #Body two
            x2_dot[i+1] = x2_dot_half[i] + self.dU_dt(t, U)[5] * delta_t/2.
            y2_dot[i+1] = y2_dot_half[i] + self.dU_dt(t, U)[7] * delta_t/2.

        return x1, y1, x2, y2

    #add in for RK2, RK4, Euler



