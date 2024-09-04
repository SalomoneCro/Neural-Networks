# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt

def Runge_Kutta(fun, interval, x_0, h, p, act_fcnt=lambda x,t,p: x):

    '''
    Runge Kutta 4th order solver
    
    Parameters:
        fun: function equal to f'
        interval: interval where the solution will be calculated
        x_0: initial value of the solution
        h: size of the steps of the method
        p: parameters of the function fun
        act_fcnt: activation function 

    Returns:
        sol: list of points of the solution to the ODE in the specified
             interval
    '''

    x_i = x_0
    t_i = interval[0]
    its =  int(np.abs(interval[0] - interval[1]) / np.abs(h))
    sol = [x_0]

    for _ in range(its - 1):

        F1 = h*fun(p, t_i, x_i)
        F2 = h*fun(p, t_i + h/2, x_i + F1/2)
        F3 = h*fun(p, t_i + h/2, x_i + F2/2)
        F4 = h*fun(p, t_i + h, x_i + F3)
        x_i += (F1 + 2*F2 + 2*F3 + F4)/6
        t_i += h
        sol.append(act_fcnt(x_i, t_i, p))
    return sol