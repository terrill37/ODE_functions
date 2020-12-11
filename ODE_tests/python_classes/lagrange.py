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
        
        energy_rel = self.energy(x1, x2, y1, y2, 
                                 x1_dot, x2_dot, y1_dot, x2_dot)

        return x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot, energy_rel

    #add in for RK2, RK4, Euler
    def Euler(self, t_pts,
              x1_0, x1_0_dot, y1_0, y1_0_dot,
              x2_0, x2_0_dot, y2_0, y2_0_dot):

        delta_t = t_pts[1] - t_pts[0]
        num_t_pts = len(t_pts)

        #initial conditions
        x1=[x1_0]
        x2=[x2_0]
        y1=[y1_0]
        y2=[y2_0]
         
        x1_dot=[x1_0_dot]
        x2_dot=[x2_0_dot]
        y1_dot=[y2_0_dot]
        y2_dot=[y2_0_dot]

        for i in range(0, num_t_pts - 1, 1):
            t = t_pts[i]

            #positional vector
            U = [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]

            #body 1
            x1_next = x1[i] + self.dU_dt(t,U)[0] * delta_t
            x1.append(x1_next)
            y1_next = y1[i] + self.dU_dt(t,U)[2] * delta_t
            y1.append(y1_next)

            x1_dot_next = x1_dot[i] + self.dU_dt(t,U)[1] * delta_t
            x1_dot.append(x1_dot_next)
            y1_dot_next = y1_dot[i] + self.dU_dt(t,U)[3] * delta_t
            y1_dot.append(y1_dot_next)

            #body 2
            x2_next = x2[i] + self.dU_dt(t,U)[4] * delta_t
            x2.append(x2_next)
            y2_next = y2[i] + self.dU_dt(t,U)[6] * delta_t
            y2.append(y1_next)

            x2_dot_next = x2_dot[i] + self.dU_dt(t,U)[5] * delta_t
            x2_dot.append(x2_dot_next)
            y2_dot_next = y2_dot[i] + self.dU_dt(t,U)[7] * delta_t
            y2_dot.append(y2_dot_next)
        
        energy_rel = self.energy(x1, x2, y1, y2, 
                                 x1_dot, x2_dot, y1_dot, x2_dot)

        return x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot, energy_rel 
 
    def RK2(self, t_pts,
            x1_0, x1_0_dot, y1_0, y1_0_dot,
            x2_0, x2_0_dot, y2_0, y2_0_dot):
        
        delta_t = t_pts[1] - t_pts[0]
        num_t_pts = len(t_pts)

        #initial conditions
        x1=[x1_0]
        x2=[x2_0]
        y1=[y1_0]
        y2=[y2_0]
         
        x1_dot=[x1_0_dot]
        x2_dot=[x2_0_dot]
        y1_dot=[y2_0_dot]
        y2_dot=[y2_0_dot]

        for i in range(0, num_t_pts - 1, 1):
            t = t_pts[i]

            #positional vector
            U = [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]
            
            #K1's
            K1 = [delta_t * x for x in self.dU_dt(t,U)]

            #K1_x1 = delta_t * self.dU_dt(t,U)[0]
            #K1_y1 = delta_t * self.dU_dt(t,U)[2]
            #K1_x2 = delta_t * self.dU_dt(t,U)[4]
            #K1_y2 = delta_t * self.dU_dt(t,U)[6]

            #K1_x1_dot = delta_t * self.dU_dt(t,U)[1]
            #K1_y1_dot = delta_t * self.dU_dt(t,U)[3]
            #K1_x2_dot = delta_t * self.dU_dt(t,U)[5]
            #K1_y2_dot = delta_t * self.dU_dt(t,U)[7]

            #update U and t
            t2 = t/2
            #K1 = dU/dt (U1(t0), t0) * h
            #U1(t0 + h) = y*(t0) + K1
            #K2 = dU/dt(U1(t0+h), t0+h)
            #U*(t+h) = U1*(t0) + h/2 (K1+K2) 
            
            #U1 = []  
            #for j in range(0, len(U)):
             #   U1.append(U[j] + K1[j])
            U0= [x1[i]+delta_t*K1[0], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]] 
            
            U1= [x1[i], x1_dot[i]+delta_t*K1[1], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]
            
            U2= [x1[i], x1_dot[i], y1[i]+delta_t*K1[2], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]
            
            U3= [x1[i], x1_dot[i], y1[i], y1_dot[i]+delta_t*K1[3],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]]
            
            U4= [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i]+delta_t*K1[4], x2_dot[i], y2[i], y2_dot[i]]
            
            U5= [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i]+delta_t*K1[5], y2[i], y2_dot[i]]
            
            U6= [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i]+delta_t*K1[6], y2_dot[i]]
            
            U7= [x1[i], x1_dot[i], y1[i], y1_dot[i],
                 x2[i], x2_dot[i], y2[i], y2_dot[i]+delta_t*K1[7]]
            
            #U = [x1[i] + K1_x1, x1_dot[i] + K1_x1_dot, 
             #    y1[i] + K1_y1, y1_dot[i] + K1_y1_dot,
              #   x2[i] + K1_x2, x2_dot[i] + K1_x2_dot, 
               #  y2[i] + K1_y2, y2_dot[i] + K1_y2_dot]
            
            #K2's
            #K2 = [delta_t*y for y in self.dU_dt(t+delta_t, U1)]

            K2_x1 = delta_t * self.dU_dt(t,U0)[0]
            K2_y1 = delta_t * self.dU_dt(t,U2)[2]
            K2_x2 = delta_t * self.dU_dt(t,U4)[4]
            K2_y2 = delta_t * self.dU_dt(t,U6)[6]

            K2_x1_dot = delta_t * self.dU_dt(t,U1)[1]
            K2_y1_dot = delta_t * self.dU_dt(t,U3)[3]
            K2_x2_dot = delta_t * self.dU_dt(t,U5)[5]
            K2_y2_dot = delta_t * self.dU_dt(t,U7)[7]

            #positions
            x1_next = x1[i]+0.5*(K1[0]+K2_x1)
            x1.append(x1_next)
            y1_next = y1[i]+0.5*(K1[2]+K2_y1)
            y1.append(y1_next)

            x2_next = x2[i]+0.5*(K1[4]+K2_x2)
            x2.append(x2_next)
            y2_next = y2[i]+0.5*(K1[6]+K2_y2)
            y2.append(y2_next)

            #velocities
            x1_dot_next = x1_dot[i]+0.5*(K1[1]+K2_x1_dot)
            x1_dot.append(x1_dot_next)             
            y1_dot_next = y1_dot[i]+0.5*(K1[3]+K2_y1_dot)
            y1_dot.append(y1_dot_next)             
                                                   
            x2_dot_next = x2_dot[i]+0.5*(K1[5]+K2_x2_dot)
            x2_dot.append(x2_dot_next)             
            y2_dot_next = y2_dot[i]+0.5*(K1[7]+K2_y2_dot)
            y2_dot.append(y2_dot_next)
        
        #relative energy
        energy_rel = self.energy(x1, x2, y1, y2, 
                                 x1_dot, x2_dot, y1_dot, x2_dot)

        return x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot, energy_rel
    
    def RK23(self, t_pts,
             x1_0, x1_0_dot, y1_0, y1_0_dot,
             x2_0, x2_0_dot, y2_0, y2_0_dot):

        U_0 = [x1_0, x1_0_dot, y1_0, y1_0_dot,
                    x2_0, x2_0_dot, y2_0, y2_0_dot]

        solution = solve_ivp(self.dU_dt, (t_pts[0], t_pts[-1]),
                             U_0, t_eval = t_pts, atol=1e-3, rtol=1e-2,
                             method='RK23')
        x1, x1_dot, y1, y1_dot, x2, x2_dot, y2, y2_dot = solution.y
        
        energy_rel = self.energy(x1, x2, y1, y2,
                                 x1_dot, x2_dot, y1_dot, x2_dot)
                                 
        return x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot, energy_rel


    def energy(self, 
               x_1, x_2, y_1, y_2,
               x_dot_1, x_dot_2, y_dot_1, y_dot_2):
        
        r_sq = ((x_1[0] - x_2[0])**2 + (y_1[0] - y_2[0])**2)       
        U = -self.G * (self.m_1 * self.m_2)/ (r_sq**(1/2))
        T = (self.m_1/2 * (x_dot_1[0]**2 + y_dot_1[0]**2) + \
             self.m_2/2 * (x_dot_2[0]**2 + y_dot_2[0]**2))
        E_0 = T + U

        energy_rel = []
        
        for i in range(0,len(x_1),1):

            r_sq = ((x_1[i] - x_2[i])**2 + (y_1[i] - y_2[i])**2)
            U = -self.G * (self.m_1 * self.m_2)/ (r_sq**(1/2))
            T = (self.m_1/2 * (x_dot_1[i]**2 + y_dot_1[i]**2) + \
                 self.m_2/2 * (x_dot_2[i]**2 + y_dot_2[i]**2))

            tot_energy = T + U
            energy_rel.append(np.abs((tot_energy-E_0)/E_0))
        
        return energy_rel


