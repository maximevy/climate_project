import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

plt.close('all')

t_year = 200 #total time of experiment [years]
t_sec = t_year*365*24*60*60 
t_tot = (0, t_sec) 

T0 = 0  #initial temperature increase condition [C]
F = 10 #forcing [W*m**(-2)]



#temperature increase differential equation with fifth order term but without noise

def dT(t,T):
    '''
    Expression of the derivative of the global annual mean of surface temperature increase
    caused by the radiative forcing. Fifth order included. Noise not included.
    '''
    alpha = 0.058 #the feedback temperature dependence [W*m**(-2)*K**(-2)] 
    beta = - 4*10**(-6) #reequilibration term [W*m**(-2)K**(-5)] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W*m**(-2)*K**(-1)] 
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
    plt.plot(sol.t/(365*24*60*60), sol.y[0], color = "firebrick")
    plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
    plt.xlabel("Time [years]")
    plt.ylabel("Temperature increase [K]")
    plt.show()

  
    
def dT_flux_plot():
    '''
    Plots the annual mean net top of atmosphere energy flux against 
    the annual mean of surface temperature.
    '''
    T_range = np.linspace(0, 25, 500) 
    alpha = 0.058 #the feedback temperature dependence [W*m**(-2)*K**(-2)] 
    beta = - 4*10**(-6) #reequilibration term [W*m**(-2)K**(-5)] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W*m**(-2)*K**(-1)] 
    net_flux = F + lbda*T_range + alpha*T_range**2 + beta*T_range**5
    plt.figure()
    plt.plot(T_range, net_flux, label='Net Flux $N$')
    plt.axhline(0, color='black', linestyle='--') 
    # plt.title("The energy Flux against the temperature increase")
    plt.xlabel("Temperature Increase [K]")
    plt.ylabel("Net Flux [W $m^{-2}$]")
    plt.grid()
    plt.show()

#first plot in report

# fig, ax = plt.subplots()

# F = 2
# sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
# ax.plot(sol.t/(365*24*60*60), sol.y[0], label = r"F = 2 $W/m^2$")

# F = 10
# sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
# ax.plot(sol.t/(365*24*60*60), sol.y[0], label = r"F = 10 $W/m^2$")

# F = 20
# sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
# ax.plot(sol.t/(365*24*60*60), sol.y[0], label = r"F = 20 $W/m^2$")

# # ax.set_title("Times series of the temperature increase for differents forcings")
# ax.set_xlabel("Temperature Increase [K]")
# ax.set_ylabel("Time [year]")
# ax.legend()
# plt.show()

#second plot in report


#temperature increase differential equation with fifth order term and noise

def dT_stoch(t, T):
    '''
    Expression of the derivative of the global annual mean of surface temperature
    caused by the radiative forcing. Fifth order included. Noise included.
    '''
    alpha = 0.058 #the feedback temperature dependence [W*m**(-2)*K**(-2)] 
    beta = - 4*10**(-6) #reequilibration term [W*m**(-2)K**(-5)] 
    lbda = - 0.88 #the slope of the top-of-atm flux N [W*m**(-2)*K**(-1)] 
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
    plt.plot(sol2.t/(365*24*60*60), sol2.y[0], color = "royalblue")
    plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
    plt.xlabel("Time [years]")
    plt.ylabel("Temperature increase [K]")
    plt.show()
    
def dT_stoch_plot2():
    '''
    Plots the time series of the annual mean of surface temperature increase
    from the dT_stoch equation. And it also plots for reference the real non 
    noisy plot.
    '''
    sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    sol2 = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    plt.figure()
    plt.plot(sol.t/(365*24*60*60), sol.y[0], color = "firebrick")
    plt.plot(sol2.t/(365*24*60*60), sol2.y[0], color = "royalblue")
    plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
    plt.xlabel("Time [years]")
    plt.ylabel("Temperature increase [K]")
    plt.show()
        
    

    
def dT_stoch_plot20():
    '''
    Plots 20 different stochastic time series of the annual mean oof surface temperature.
    '''
    sol = solve_ivp(fun = dT, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
    i = 0
    while i < 20:
        sol2 = solve_ivp(fun = dT_stoch, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))
        plt.plot(sol2.t/(365*24*60*60), sol2.y[0])
        plt.title(f"20 Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
        plt.xlabel("Time [years]")
        plt.ylabel("Temperature increase [K]")
        i += 1
    plt.plot(sol.t/(365*24*60*60), sol.y[0], color = "firebrick")

    
# dT_plot()
# dT_stoch_plot()
dT_flux_plot()
# dT_stoch_plot2()
# dT_stoch_plot20()











