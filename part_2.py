import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

# plt.close('all')

t_year = 20000
t_sec = t_year*365*24*60*60 
t_tot = (0, t_sec) #Important : find the true timescales for computation

T0 = 0  #initial temperature condition [K]
F = 2 #forcing [W*m**(-2)]

# temp increase diff equation with fifth order and noise



def dT_stoch(t, T):
    '''
    Expression of the derivative of the global annual mean of surface temperature
    caused by the radiative forcing.
    '''
    alpha = 0.058 #the feedback temperature dependence [W/m**2/K] 
    beta = - 4*10**(-6) # [W/m**2/K] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W/m**2/K] 
    c = 8.36*10**8 #J*K**(-1)*m**(-2)
    noise = random.gauss(0, 10) #normal is 10
    dT = 1/c*(F + lbda*T + alpha*T**2 + beta*T**5 )# noise)
    return dT



sol = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))

plt.figure()
plt.plot(sol.t/(365*24*60*60), sol.y[0])
plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
plt.xlabel("Time [years]")
plt.ylabel("Temperature difference [K]")
plt.show()

dT_real = dT_stoch(sol.t, sol.y[0])
# print(dT_real)
# print(dT_real*8.36*10**8)

plt.figure()
plt.grid()
plt.plot(273.15 + sol.y[0], dT_real*8.36*10**8)
plt.show()




# plot of 20 different solutions

# i = 0
# while i < 20:
#     sol = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
#     plt.plot(sol.t/(365*24*60*60), sol.y[0])
#     plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
#     plt.xlabel("Time [years]")
#     plt.ylabel("Temperature difference [K]")
#     i += 1


#Important thing to notice. System is nonlinear. When forcing around 3.5
#there is a strong increase in the final temperature increase of the system.















