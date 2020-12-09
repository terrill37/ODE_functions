import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy import linalg as LA

from mpmath import *

import sympy as sym
from sympy import *

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
                                          (self.q[j],p[j])])

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

