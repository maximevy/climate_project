import numpy as np
from scipy.integrate import solve_ivp

N = 2000
noise_std = 0.02
noise = np.random.normal(0, noise_std, size = N)

t_span = (0, 1000)
t_eval = np.linspace(*t_span, 2000000)

T0 = 0

def dT_stoch(T):
    alpha = 0.058 #W/m**2/K 
    beta = 4*10**(-6) #W/m**2/K 
    lbda = - 0.88 #W/m**2/K 
    F = 0
    c = 8.36*108 #J*K**(-1)*m**(-2)
    dT = 1/c*(F + lbda*T + alpha*T**2 + beta*T**5) #+ noise)
    return dT



solve_ivp(dT_stoch, t_span, T0, t_eval = t_eval)























