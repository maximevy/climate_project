import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# t_year = 200
# t_sec = t_year*365*24*60*60 
t_sec = 10000
t_tot = (0, t_sec) #Important : find the true timescales for computation
# t_eval = np.linspace(*t_tot, 2000000)

T0 = 0 #initial temperature condition

def dT_stoch(t, T):
    '''
    Expression of the derivative of the global annual mean of surface temperature
    caused by the radiative forcing.
    '''
    alpha = 0.058 #W/m**2/K 
    beta = 4*10**(-6) #W/m**2/K 
    lbda = - 0.88 #W/m**2/K 
    F = 0 #W*m**{-2}
    c = 8.36*108 #J*K**(-1)*m**(-2)
    dT = 1/c*(F + lbda*T + alpha*T**2 + beta*T**5 + np.random.normal(0, 0.1, size = 1) )
    return dT


# print(dT_stoch(2))

sol = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))


plt.plot(sol.t, sol.y[0])
plt.xlabel("Time")
plt.ylabel("Temperature difference")



















