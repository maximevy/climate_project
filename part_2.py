import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

plt.close('all')

t_year = 2000 #total time of experiment [years]
t_sec = t_year*365*24*60*60 
t_tot = (0, t_sec) 

T0 = 0  #initial temperature condition [C]
F = 3.6 #forcing [W*m**(-2)]



#temperature increase differential equation with fifth order term but without noise

def dT(t,T):
    '''
    Expression of the derivative of the global annual mean of surface temperature increase
    caused by the radiative forcing. Fifth order included. Noise not included.
    '''
    alpha = 0.058 #the feedback temperature dependence [W/m**2/K] 
    beta = - 4*10**(-6) #reequilibration term [W/m**2/K] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W/m**2/K] 
    c = 8.36*10**8 #heat capacity [J*K**(-1)*m**(-2)]
    dT = 1/c*(F + lbda*T + alpha*T**2 + beta*T**5)
    return dT


def dT_plot():
    '''
    Plots the time series of the annual mean of surface temperature increase
    from the dT equation.
    '''
    sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    plt.figure()
    plt.plot(sol.t/(365*24*60*60), sol.y[0])
    plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
    plt.xlabel("Time [years]")
    plt.ylabel("Temperature difference [K]")
    plt.show()


def dT_flux_plot():
    '''
    Plots the annual mean net top of atmosphere energy flux against 
    the annual mean of surface temperature from the dT equation.
    '''
    sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    dT_real = dT(sol.t, sol.y[0])
    plt.figure()
    plt.grid()
    plt.plot(273.15 + sol.y[0], dT_real*8.36*10**8)
    plt.title("Annual means of temperature against the top of atmosphere energy flux")
    plt.xlabel("Temperature [K]")
    plt.ylabel(r'Flux [$Jm^{-2}$]')
    plt.show()




#temperature increase differential equation with fifth order term and noise

def dT_stoch(t, T):
    '''
    Expression of the derivative of the global annual mean of surface temperature
    caused by the radiative forcing. Fifth order included. Noise included.
    '''
    alpha = 0.058 #the feedback temperature dependence [W/m**2/K] 
    beta = - 4*10**(-6) #reequilibration term [W/m**2/K] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W/m**2/K] 
    c = 8.36*10**8 #heat capacity [J*K**(-1)*m**(-2)]
    noise = random.gauss(0, 10) #ideal value for the std is 10
    dT = 1/c*(F + lbda*T + alpha*T**2 + beta*T**5 + noise)
    return dT


def dT_stoch_plot():
    '''
    Plots the time series of the annual mean of surface temperature increase
    from the dT_stoch equation.
    '''
    sol2 = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    plt.figure()
    plt.plot(sol2.t/(365*24*60*60), sol2.y[0])
    plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
    plt.xlabel("Time [years]")
    plt.ylabel("Temperature difference [K]")
    plt.show()
    
    
def dT_stoch_plot20():
    '''
    Plots 20 different stochastic time series of the annual mean oof surface temperature.
    '''
    i = 0
    while i < 20:
        sol2 = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
        plt.plot(sol2.t/(365*24*60*60), sol2.y[0])
        plt.title(f"20 Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
        plt.xlabel("Time [years]")
        plt.ylabel("Temperature difference [K]")
        i += 1
    

    
dT_plot()
dT_flux_plot()
# dT_stoch_plot()














