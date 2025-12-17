import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
We solve the equations 1 and 2 of the assignment and reproduce the figures 1a, 1d, 2a and 2b 
of Bloch-Johnson et al. (2015)
"""

# Constants 
F2x = 3.71

def Fnx(n):
    return np.log2(n)*3.71

# Figure 1a 
lmbda = -0.88
ac = -0.035
am = 0.03
ah = 0.058
T0 = 287

def N_T(a, b, c, start, stop, steps):
    x = np.linspace(start - T0, stop - T0, num=steps)
    y = a*x**2 + lmbda*x + c
    roots = np.roots([a, b, c])
    return [y, roots]

def plot_1a(): # plots figure 1a of the paper 
    x = np.linspace(285, 297, 100)
    linear = N_T(0, lmbda, F2x, 285, 297, 100)
    f_c = N_T(ac, lmbda, F2x, 285, 297, 100)
    f_m = N_T(am, lmbda, F2x, 285, 297, 100) 
    f_h = N_T(ah, lmbda, F2x, 285, 297, 100)
    
    fig, ax = plt.subplots()
    ax.plot(x, linear[0], linestyle = "--", color = "k")
    ax.plot(x, f_c[0], color = "b")
    ax.plot(x, f_m[0], color = "g")
    ax.plot(x, f_h[0], color = "r")
    
    # plot slope of lambda
    xmin, xmax = 286, 288
    mask = (x >= xmin) & (x <= xmax)
    ax.plot(x[mask], linear[0][mask], color = "m")
    ax.text(287, 4, r"$\lambda$", color = "m")
    
    # text and annotations 
    ax.text(286.75, -0.7, r"$T_0$")

    
    ax.set_xlabel("T (K)")
    ax.set_ylabel("N (W/m²)")
    ax.set_title("Figure 1a")
    
    ax.set_ylim(-2, 8)
    ax.set_xlim(285, 297)
    
    ax.set_xticks([285, 287, 289, 291, 293, 295, 297]) # set ticks 
    ax.axhline(0, color = "k") # add horizontal line at y=0 
    for xi in [linear[1][-1], f_c[1][-1], f_m[1][-1]]: # add markers at roots on horizontal line 
        ax.plot([xi + T0, xi + T0], [-0.15, 0.15], color='k')
        ax.plot([287, 287], [-0.15, 0.15], color = "k")
    
    plt.tight_layout()
    plt.show()
 
plot_1a()





# temp increase diff equation without fifth order and noise

t_year = 200
t_sec = t_year*365*24*60*60 
t_tot = (0, t_sec) #Important : find the true timescales for computation

T0 = 0  #initial temperature condition
F = 2 #forcing [W*m**(-2)]

# def dT_basic(t, T):
#     '''
#     Expression of the derivative of the global annual mean of surface temperature
#     caused by the radiative forcing.
#     '''
#     alpha = 0.058 #the feedback temperature dependence [W/m**2/K] 
#     beta = - 4*10**(-6) # [W/m**2/K] 
#     lbda = - 0.88 #the slope of the top-of-atm flux N [W/m**2/K] 
#     c = 8.36*10**8 #J*K**(-1)*m**(-2)
#     dT = 1/c*(F + lbda*T + alpha*T**2)
#     return dT


# sol = solve_ivp(fun = dT_basic, t_span = t_tot, y0 = [T0], t_eval = np.linspace(*t_tot, 20000))


# plt.plot(sol.t/(365*24*60*60), sol.y[0])
# plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
# plt.xlabel("Time [years]")
# plt.ylabel("Temperature difference [K]")





