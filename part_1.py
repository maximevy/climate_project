import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

"""
We solve the equations 1 and 2 of the assignment and reproduce the figures
1a, 1d, 2a and 2b of Bloch-Johnson et al. (2015)
"""

# Constants 
F2X = 3.71
T0 = 287

def Fnx(n): 
    """ Return the radiative forcing for a nx increase of the C02 concentration
    in the atmosphere.
    """
    return np.log2(n)*3.71

# Figure 1a 
def N_T(a, b, c, start, stop, steps):
    """ Solve equation 1 in the paper and return N as a function of T and
    the roots (Delta_T).
    """
    x = np.linspace(start - T0, stop - T0, num=steps)
    y = a*x**2 + b*x + c
    roots = np.roots([a, b, c])
    return [y, roots]

def fig_1a(): 
    """ Generate figure 1a of the paper."""
    # Data
    lbda = -0.88 # feedback parameter 
    ac = -0.035   # feedback temperature dependence parameter
    am = 0.03
    ah = 0.058
    x = np.linspace(285, 297, 100)
    linear = N_T(0, lbda, F2X, 285, 297, 100)
    f_c = N_T(ac, lbda, F2X, 285, 297, 100)
    f_m = N_T(am, lbda, F2X, 285, 297, 100) 
    f_h = N_T(ah, lbda, F2X, 285, 297, 100)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(x, linear[0], linestyle = "--", color = "k",
            label = f"linear, $\Delta T_{{2x}}={linear[1][0]:.1f}K$"
            )
    ax.plot(x, f_c[0], color = "b",
            label = f"$a_C = -0.035, \Delta T_{{2x}} = {f_c[1][1]:.1f}K$"
            )
    ax.plot(x, f_m[0], color = "g",
            label = f"$a_M = 0.03, \Delta T_{{2x}}= {f_m[1][1]:.1f}K$"
            )
    ax.plot(x, f_h[0], color = "r",
            label = r"$a_H=0.058, \Delta T_{{2x}}= ? $"
            )
    
    # plot slope of lambda
    xmin, xmax = 286.25, 287.75
    mask = (x >= xmin) & (x <= xmax)
    ax.plot(x[mask], linear[0][mask], color = "magenta", lw=2.5)
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
                arrowprops = dict(facecolor = "red", arrowstyle="->", lw=1)
)

    # labels 
    ax.set_xlabel("T (K)")
    ax.set_ylabel("N (W/m²)")
    #ax.set_title("Figure 1a")
    
    # axis limits
    ax.set_ylim(-2, 8)
    ax.set_xlim(285, 297)
    
    # ticks 
    ax.set_xticks([285, 287, 289, 291, 293, 295, 297]) # set ticks 
    ax.axhline(0, color = "k") # add horizontal line at y=0 
    for xi in [linear[1][-1], f_c[1][-1], f_m[1][-1]]: # add markers at roots 
                                                       # on horizontal line 
        ax.plot([xi + T0, xi + T0], [-0.15, 0.15], color='k')
        ax.plot([287, 287], [-0.15, 0.15], color = "k")

    # legend
    handles, labels = ax.get_legend_handles_labels()
    proxy = Line2D([0], [0], color = "none", linestyle="")
    handles.insert(0, proxy)
    labels.insert(0, r"$\lambda=-0.88\ W/m^2/K$")
    ax.legend(handles, labels)
    
    plt.tight_layout()
    plt.savefig("figure_1a.pdf")
    plt.show()


# figure 1d
def fig_1d():
    """ generate figure 1d from the paper."""
    # data 
    lbda = -1.28 # feedback parameter
    a_h = 0.058 # feedback temperature difference parameter
    F4x = Fnx(4) # radiative forcing for 4x CO2 increase
    x = np.linspace(286, 300, 100)
    lin_2x = N_T(0, lbda, F2X, 286, 300, 100)
    quad_2x = N_T(a_h, lbda, F2X, 286, 300, 100)
    lin_4x = N_T(0, lbda, F4x, 286, 300, 100)
    quad_4x = N_T(a_h, lbda, F4x, 286, 300, 100)
    linF1 = N_T(0, lbda, 0.5, 286, 300, 100)
    quadF1 = N_T(a_h, lbda, 0.5, 286, 300, 100)
    linF2 = N_T(0, lbda, 2, 286, 300, 100)
    quadF2 = N_T(a_h, lbda, 2, 286, 300, 100)  
    
    # plotting
    fig, ax = plt.subplots()
    ax.plot(x, lin_2x[0], linestyle = "--", color = "b",
            label = f"$2xCO_2\ (linear), \Delta T_{{2x}}={lin_2x[1][-1]:.1f}K$"
            )
    ax.plot(x, quad_2x[0], linestyle = "-", color = "b",
            label = f"$2xCO_2\ (quad), \Delta T_{{2x}}={quad_2x[1][-1]:.1f}K$"
            )
    ax.plot(x, lin_4x[0], linestyle = "--", color = "r",
            label = f"$4xCO_2\ (linear), \Delta T_{{4x}}={lin_4x[1][-1]:.1f}K$"
            )
    ax.plot(x, quad_4x[0], linestyle = "-", color = "r",
            label = r"$4xCO_2\ (quad), \Delta T_{{4x}}= ? $"
            )
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
    #ax.set_title("Figure 1d")
    
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
    plt.savefig("figure_1d.pdf")
    plt.show()
        

# figure 2
def delta_T_vs_a(lbda, n):
    """
    Delta T as a function of values of a, for a given lambda and CO2 increase
    
    Parameters
    ----------
    lbda : feedback paramater.
    n : Number of CO2 doublings.

    Returns
    -------
    delta_T : list of values of delta_T.

    """
    F = Fnx(n)
    delta_T = []
    a_values = np.linspace(-0.1, 0.1, 400)
    for a in a_values:
        root = np.roots([a, lbda, F])[-1]
        if isinstance(root, (int, float)): # replace complex values with NaN
            delta_T.append(root)           # because it means we reached 
        else:                              # runaway warming  
            delta_T.append(np.nan)
    return delta_T

# figure 2a
def fig_2a():
    """ generate figure 2a from the paper."""
    # data 
    x = np.linspace(-0.1, 0.1, 400)
    delta_T_2x_vs_a_max = delta_T_vs_a(-0.79, 2)
    delta_T_2x_vs_a_mean = delta_T_vs_a(-1.17, 2)
    delta_T_2x_vs_a_min = delta_T_vs_a(-1.78, 2)
    delta_T_2x_vs_a_0 = delta_T_vs_a(-1*(10**-20), 2)
    
    # plotting 
    fig, ax = plt.subplots()
    ax.plot(x, delta_T_2x_vs_a_0, color="k",
            label=r"$\lambda = 0\ W/m^2/K$"
            )
    ax.plot(x, delta_T_2x_vs_a_max, color="m",
            label=r"$\lambda_{max} = -0.79$"
            )
    ax.plot(x, delta_T_2x_vs_a_mean, color="g",
            label=r"$\lambda_{mean} = -1.17$"
            )
    ax.plot(x, delta_T_2x_vs_a_min, color="C1",
            label=r"$\lambda_{min} = -1.78$"
            )
    
    # limits
    ax.set_ylim(0, 10)
    
    # labels
    ax.set_ylabel(r"$\Delta T_{2x}(K)$")
    ax.set_xlabel(r"$a\ (W/m^2/K^2)$")
    ax.set_title(r"$2xCO_2$")
    
    # ticks
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    
    ax.axvline(x=0, color="k", lw=0.5) # vertical line at a = 0 
    
    # legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figure_2a.pdf")
    plt.show()


# figure 2b 
def fig_2b():
    """ generate figure 2b from the paper.""" 
    # data 
    x = np.linspace(-0.1, 0.1, 400)
    delta_T_4x_vs_a_max = delta_T_vs_a(-0.79, 4)
    delta_T_4x_vs_a_mean = delta_T_vs_a(-1.17, 4)
    delta_T_4x_vs_a_min = delta_T_vs_a(-1.78, 4)
    delta_T_4x_vs_a_0 = delta_T_vs_a(-1*(10**-20), 4)
    
    # plotting 
    fig, ax = plt.subplots()
    ax.plot(x, delta_T_4x_vs_a_0, color="k",
            label=r"$\lambda = 0\ W/m^2/K$"
            )
    ax.plot(x, delta_T_4x_vs_a_max, color="m",
            label=r"$\lambda_{max} = -0.79$"
            )
    ax.plot(x, delta_T_4x_vs_a_mean, color="g",
            label=r"$\lambda_{mean} = -1.17$"
            )
    ax.plot(x, delta_T_4x_vs_a_min, color="C1",
            label=r"$\lambda_{min} = -1.78$"
            )
    
    # limits
    ax.set_ylim(0, 20)
    
    # labels
    ax.set_ylabel(r"$\Delta T_{4x}(K)$")
    ax.set_xlabel(r"$a\ (W/m^2/K^2)$")
    ax.set_title(r"$4xCO_2$")
    
    # ticks
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    ax.set_yticks([0, 5, 10, 15, 20])
    
    ax.axvline(x=0, color="k", lw=0.5) # vertical line at a=0
    
    # legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figure_2b.pdf")
    plt.show()


# call the figure functions to plot them
#fig_1a()
#fig_1d()
#fig_2a()
#fig_2b()

# temp increase diff equation without fifth order and noise
t_year = 200
t_sec = t_year * 365 * 24 * 60 * 60 
t_tot = (0, t_sec) 
Delta_T_0 = 0  #initial temperature condition

def dT_basic(t, T, alpha, lbda, F):
   '''
   Expression of the derivative of the global annual mean of surface
   temperature caused by the radiative forcing. Fifth order not included.
   Noise not included.
   
   Parameters:
    - t: time (required by solve_ivp)
    - T: temperature array
    - alpha: quadratic feedback parameter
    - lbda: linear feedback parameter
    - F: radiative forcing
   
   '''
   C = 8.36e8  # J*K**(-1)*m**(-2)
   dT = 1/C * (F + lbda*T[0] + alpha*T[0]**2)
   return [dT]


def plot_transient_behaviour():
    # Create figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    divider = 365 * 24 * 60 * 60
    
    # plot on first subplot (2x CO2)
    lbda_values = [-0.79, -1.17, -1.78]
    lbda_names = [r"$\lambda_{max}$", r"$\lambda_{mean}$", r"$\lambda_{min}$"]
    a_values = [-0.035, 0.03, 0.058]
    a_names = [r"$a_C$", r"$a_M$", r"$a_H$"]
    
    for index_lbda, lbda in enumerate(lbda_values):
        for index_a, a in enumerate(a_values):
            sol = solve_ivp(fun=lambda t, T: dT_basic(t, T, a, lbda, Fnx(2)),
                            t_span=t_tot, y0=[Delta_T_0],
                            t_eval=np.linspace(*t_tot, 20000)
                            )
            ax1.plot(sol.t / divider, sol.y[0],
                     label=f"{lbda_names[index_lbda]} = {lbda}, {a_names[index_a]} = {a}"
                     )
            
    # Labels
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Temperature change (K)")
    ax1.set_title(r"$2xCO_2$")
    
    # Move legend outside the plot area
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.legend()
    
    # plot on second subplot (4x CO2)
    for index_lbda, lbda in enumerate(lbda_values):
        for index_a, a in enumerate(a_values):
            sol = solve_ivp(fun=lambda t, T: dT_basic(t, T, a, lbda, Fnx(4)),
                            t_span=t_tot, y0=[Delta_T_0],
                            t_eval=np.linspace(*t_tot, 20000)
                            )
            # Check if solution is valid
            if sol.success:
                ax2.plot(sol.t / divider, sol.y[0],
                         label=f"{lbda_names[index_lbda]} = {lbda}, {a_names[index_a]} = {a}"
                         )
                print(f"λ={lbda}, a={a}: max temp = {sol.y[0].max():.2e}")
            else:
                print(f"λ={lbda}, a={a}: Integration failed")
                 
    # Labels
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Temperature change (K)")
    ax2.set_title(r"$4xCO_2$")
    
    # Move legend outside the plot area
    #ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.savefig(fname="graphs/part1/transient_behaviour.pdf")
    plt.show()
    

plot_transient_behaviour()