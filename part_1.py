import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# temp increase diff equation without fifth order and noise

t_year = 200
t_sec = t_year*365*24*60*60 
t_tot = (0, t_sec) #Important : find the true timescales for computation

T0 = 0  #initial temperature condition
F = 2 #forcing [W*m**(-2)]

def dT_basic(t, T):
    '''
    Expression of the derivative of the global annual mean of surface temperature
    caused by the radiative forcing.
    '''
    alpha = 0.058 #the feedback temperature dependence [W/m**2/K] 
    beta = - 4*10**(-6) # [W/m**2/K] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W/m**2/K] 
    c = 8.36*10**8 #J*K**(-1)*m**(-2)
    dT = 1/c*(F + lbda*T + alpha*T**2)
    return dT


sol = solve_ivp(fun = dT_basic, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))


plt.plot(sol.t/(365*24*60*60), sol.y[0])
plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
plt.xlabel("Time [years]")
plt.ylabel("Temperature difference [K]")





