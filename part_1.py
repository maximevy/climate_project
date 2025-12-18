import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
We solve the equations 1 and 2 of the assignment and reproduce the figures 1a, 1d, 2a and 2b 
of Bloch-Johnson et al. (2015)
"""

# Constants 
F2x = 3.71
T0 = 287

def Fnx(n):
    return np.log2(n)*3.71

# Figure 1a 
lmbda = -0.88
ac = -0.035
am = 0.03
ah = 0.058


def N_T(a, b, c, start, stop, steps):
    x = np.linspace(start - T0, stop - T0, num=steps)
    y = a*x**2 + lmbda*x + c
    roots = np.roots([a, b, c])
    return [y, roots]

def fig_1a(): # plots figure 1a of the paper 
    # Data
    x = np.linspace(285, 297, 100)
    linear = N_T(0, lmbda, F2x, 285, 297, 100)
    f_c = N_T(ac, lmbda, F2x, 285, 297, 100)
    f_m = N_T(am, lmbda, F2x, 285, 297, 100) 
    f_h = N_T(ah, lmbda, F2x, 285, 297, 100)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(x, linear[0], linestyle = "--", color = "k", label = f"linear, $\Delta T_{{2x}}={linear[1][0]:.1f}K$")
    ax.plot(x, f_c[0], color = "b", label = f"$a_C = -0.035, \Delta T_{{2x}} = {f_c[1][1]:.1f}K$")
    ax.plot(x, f_m[0], color = "g", label = f"$a_M = 0.03, \Delta T_{{2x}}= {f_m[1][1]:.1f}K$")
    ax.plot(x, f_h[0], color = "r", label = r"$a_H=0.058, \Delta T_{{2x}}= ? $")
    
    # plot slope of lambda
    xmin, xmax = 286.25, 287.75
    mask = (x >= xmin) & (x <= xmax)
    ax.plot(x[mask], linear[0][mask], color = "magenta")
    ax.text(287, 4, r"$\lambda$", color = "magenta")
    
    # text and arrows 
    ax.text(286.75, -0.7, r"$T_0$") # T0
    ax.text(289, -0.8, r"$\Delta T_{2x}$", ha = "center") # DT2X
    ax.text(286.4, 2, r"$F_{2x}$")
    ax.annotate( # Delta T2x left to right arrow
    '',                      # empty string → just arrow
    xy=(f_c[1][1]+T0, -0.2),               # arrow tip
    xytext=(287.1, -0.2),           # arrow tail
    arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1)
)
    ax.annotate("", xy = (T0, 3.7), xytext = (T0, 0.2), 
                arrowprops = dict(facecolor = "red", arrowstyle="->", lw=1))

    # labels 
    ax.set_xlabel("T (K)")
    ax.set_ylabel("N (W/m²)")
    ax.set_title("Figure 1a")
    
    # axis limits
    ax.set_ylim(-2, 8)
    ax.set_xlim(285, 297)
    
    # ticks 
    ax.set_xticks([285, 287, 289, 291, 293, 295, 297]) # set ticks 
    ax.axhline(0, color = "k") # add horizontal line at y=0 
    for xi in [linear[1][-1], f_c[1][-1], f_m[1][-1]]: # add markers at roots on horizontal line 
        ax.plot([xi + T0, xi + T0], [-0.15, 0.15], color='k')
        ax.plot([287, 287], [-0.15, 0.15], color = "k")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    proxy = Line2D([0], [0], color = "none", linestyle="")
    handles.insert(0, proxy)
    labels.insert(0, r"$\lambda=-0.88\ W/m^2/K$")
    ax.legend(handles, labels)
    
    plt.tight_layout()
    plt.show()
 
fig_1a()

# figure 1d
lmbda = -1.28
a_h = 0.058
F4x = Fnx(4)

def fig_1d():
    # data 
    x = np.linspace(286, 300, 100)
    lin_2x = N_T(0, lmbda, F2x, 286, 300, 100)
    quad_2x = N_T(a_h, lmbda, F2x, 286, 300, 100)
    lin_4x = N_T(0, lmbda, F4x, 286, 300, 100)
    quad_4x = N_T(a_h, lmbda, F4x, 286, 300, 100)
    linF1 = N_T(0, lmbda, 0.5, 286, 300, 100)
    quadF1 = N_T(a_h, lmbda, 0.5, 286, 300, 100)
    linF2 = N_T(0, lmbda, 2, 286, 300, 100)
    quadF2 = N_T(a_h, lmbda, 2, 286, 300, 100)  
    
    # plotting
    fig, ax = plt.subplots()
    ax.plot(x, lin_2x[0], linestyle = "--", color = "b", label = f"$2xCO_2\ (linear), \Delta T_{{2x}}={lin_2x[1][-1]:.1f}K$")
    ax.plot(x, quad_2x[0], linestyle = "-", color = "b", label = f"$2xCO_2\ (quad), \Delta T_{{2x}}={quad_2x[1][-1]:.1f}K$")
    ax.plot(x, lin_4x[0], linestyle = "--", color = "r", label = f"$4xCO_2\ (linear), \Delta T_{{4x}}={lin_4x[1][-1]:.1f}K$")
    ax.plot(x, quad_4x[0], linestyle = "-", color = "r", label = r"$4xCO_2\ (quad), \Delta T_{{4x}}= ? $")
    ax.plot(x, linF1[0], linestyle = "--", color = "k")
    ax.plot(x, quadF1[0], linestyle = "-", color = "k")
    ax.plot(x, linF2[0], linestyle = "--", color = "k")
    ax.plot(x, quadF2[0], linestyle = "-", color = "k")
    
    # plot slope of lambda
    xmin, xmax = 286.8, 287.2
    mask = (x >= xmin) & (x <= xmax)
    ax.plot(x[mask], linF2[0][mask], color = "magenta", lw = 2.5)

    # labels 
    ax.set_xlabel("T (K)")
    ax.set_ylabel("N (W/m²)")
    ax.set_title("Figure 1d")
    
    # annotations 
    ax.axhline(0, color = "k") # add horizontal line at y=0 
    ax.plot([287, 287], [-0.15, 0.15], color = "k") # small tick at T0
    ax.text(T0, -1.5, r"$T_0$", ha = "center")
    
    # axis limits
    ax.set_ylim(-7, 21)
    
    # legend
    handles, labels = ax.get_legend_handles_labels()
    proxy = Line2D([0], [0], color = "none", linestyle="")
    for i in range(2):
        handles.insert(0, proxy)
    labels.insert(0, r"$a_H = 0.058\ W/m^2/K^2$")
    labels.insert(0, r"$\lambda=-1.28\ W/m^2/K$")
    
    ax.legend(handles, labels, loc = "upper right")
    
    plt.tight_layout()
    plt.show()
        
fig_1d()

# figure 2 
# 2a
def delta_T_vs_a(lmbda, n):
    """
    Delta T as a function of values of a, for a given lambda and CO2 increase
    
    Parameters
    ----------
    lmbda : feedback paramater.
    n : Number of CO2 doublings.

    Returns
    -------
    delta_T : list of values of delta_T.

    """
    F = Fnx(n)
    delta_T = []
    a_values = np.linspace(-0.1, 0.1, 400)
    for a in a_values:
        root = np.roots([a, lmbda, F])[-1]
        if isinstance(root, (int, float)):
            delta_T.append(root)
        else: 
            delta_T.append(np.nan)
    return delta_T



def fig_2a():
    # data 
    x = np.linspace(-0.1, 0.1, 400)
    delta_T_2x_vs_a_max = delta_T_vs_a(-0.79, 2)
    delta_T_2x_vs_a_mean = delta_T_vs_a(-1.17, 2)
    delta_T_2x_vs_a_min = delta_T_vs_a(-1.78, 2)
    delta_T_2x_vs_a_0 = delta_T_vs_a(-1*(10**-20), 2)
    
    # plotting 
    fig, ax = plt.subplots()
    ax.plot(x, delta_T_2x_vs_a_0, color="k", label=r"$\lambda = 0\ W/m^2/K$")
    ax.plot(x, delta_T_2x_vs_a_max, color="m", label=r"$\lambda_{max} = -0.79$")
    ax.plot(x, delta_T_2x_vs_a_mean, color="g", label=r"$\lambda_{mean} = -1.17$")
    ax.plot(x, delta_T_2x_vs_a_min, color="C1", label=r"$\lambda_{min} = -1.78$")
    
    # limits
    ax.set_ylim(0, 10)
    
    # labels
    ax.set_ylabel(r"$\Delta T_{2x}(K)$")
    ax.set_xlabel(r"$a\ (W/m^2/K^2)$")
    ax.set_title(r"$2xCO_2$")
    
    # ticks
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    
    ax.axvline(x=0, color="k", lw=0.5)
    
    # legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
fig_2a()

def fig_2b():
    # data 
    x = np.linspace(-0.1, 0.1, 400)
    delta_T_4x_vs_a_max = delta_T_vs_a(-0.79, 4)
    delta_T_4x_vs_a_mean = delta_T_vs_a(-1.17, 4)
    delta_T_4x_vs_a_min = delta_T_vs_a(-1.78, 4)
    delta_T_4x_vs_a_0 = delta_T_vs_a(-1*(10**-20), 4)
    
    # plotting 
    fig, ax = plt.subplots()
    ax.plot(x, delta_T_4x_vs_a_0, color="k", label=r"$\lambda = 0\ W/m^2/K$")
    ax.plot(x, delta_T_4x_vs_a_max, color="m", label=r"$\lambda_{max} = -0.79$")
    ax.plot(x, delta_T_4x_vs_a_mean, color="g", label=r"$\lambda_{mean} = -1.17$")
    ax.plot(x, delta_T_4x_vs_a_min, color="C1", label=r"$\lambda_{min} = -1.78$")
    
    # limits
    ax.set_ylim(0, 20)
    
    # labels
    ax.set_ylabel(r"$\Delta T_{4x}(K)$")
    ax.set_xlabel(r"$a\ (W/m^2/K^2)$")
    ax.set_title(r"$4xCO_2$")
    
    # ticks
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    ax.set_yticks([0, 5, 10, 15, 20])
    
    ax.axvline(x=0, color="k", lw=0.5)
    
    # legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
fig_2b()
         

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

"""
plt.plot(sol.t/(365*24*60*60), sol.y[0])
plt.title(f"Time series of the increase in global annual mean surface temperature with a forcing of F = {F}")
plt.xlabel("Time [years]")
plt.ylabel("Temperature difference [K]")
"""