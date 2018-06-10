"""
Finds the relationship between burn starting height, burn time, final mass and
thrust for the simple 1-d rocket landing problem without drag.

D. Malyuta -- ACL, University of Washington
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

import progressbar # sudo -H pip install progressbar2
from rocket import Rocket

# Progress bar appearance
pb_widgets=[progressbar.Percentage(),' ',
            progressbar.Bar(),' (',progressbar.ETA(),')']

# LaTeX font in plots
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font',**{'family':'serif'})
matplotlib.rc('text', usetex=True)

def simulateForHeight(rocket,H_1):
    """
    Simulate the rocket landing problem for a particular burn start height H_1.
    
    Parameters
    ----------
    H_1 : float
        Burn start height.
        
    Returns
    -------
    fuel_used : float
        Amount of fuel used.
    burn_thrust : float
        Constant thrust value during the landing burn.
    burn_time : float
        Duration of landing burn.
    time_history : array
        Simulation times.
    height_history : array
        Simulation rocket heights.
    """
    rocket.setLandingBurnStartHeight(H_1)
    
    # Simulate the dynamics
    t_f = 100.
    x_0 = np.array([rocket.H_0,rocket.v_0,rocket.m_0, # Ideal system
                    rocket.H_0,rocket.v_0,rocket.m_0 # Real system
                    ])
    
    state = solve_ivp(fun = rocket.dynamics,
                      t_span = (0,t_f),
                      y0 = x_0,
                      events = [rocket.burnEvent,rocket.ascentEvent],
                      max_step = 0.001)
    
    # Extract simulation results
    t = state.t
    h = state.y[3]
    m = state.y[5]
    
    fuel_used = m[0]-m[-1]
    burn_thrust = rocket.T
    burn_time = rocket.t_b
    time_history = t
    height_history = h
    
    return fuel_used,burn_thrust,burn_time,time_history,height_history

burn_start_heights = np.linspace(100,300,20)
fuel_used,burn_thrust,burn_time,sim_time,sim_height = [],[],[],[],[]
for i in progressbar.progressbar(range(len(burn_start_heights)), widgets=pb_widgets):
    rocket = Rocket(noisy=False,controlled=False)
    _fuel,_burn_thrust,_burn_time,_sim_t,_sim_h = simulateForHeight(rocket,burn_start_heights[i])
    fuel_used.append(_fuel)
    burn_thrust.append(_burn_thrust)
    burn_time.append(_burn_time)
    sim_time.append(_sim_t)
    sim_height.append(_sim_h)

#%% Plot result
    
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
for i in range(len(burn_start_heights)):
    ax.plot(sim_time[i],sim_height[i],label=("%d"%(burn_start_heights[i])))
ax.set_ylabel('Height $h$ [m]')
ax.set_xlabel('Time $t$ [s]')
#ax.legend()
plt.autoscale(tight=True)
plt.tight_layout()
plt.show()
fig.savefig('media/height_history.png',
            bbox_inches='tight', format='png', transparent=True, dpi=150)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig = plt.figure(2)
plt.clf()
ax1 = fig.add_subplot(111)
ax1.plot(burn_start_heights,fuel_used,color=colors[0])
ax1.set_ylabel('Fuel used $m_0-m_{\\mathrm{f}}$ [kg]')
ax1.spines['right'].set_color(colors[1]) 
ax1.spines['left'].set_color(colors[0]) 
ax1.tick_params(axis='y', colors=colors[0])
ax1.yaxis.label.set_color(colors[0])
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
ax2.plot(burn_start_heights,np.array(burn_thrust)/1e3,color=colors[1])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors=colors[1])
ax2.yaxis.label.set_color(colors[1])
ax2.set_ylabel('Thrust $T$ [kN]')
ax1.set_xlabel('Burn start height $H_1$ [m]')
plt.autoscale(tight=True)
plt.tight_layout()
plt.show()
fig.savefig('media/fuel_used.png',
            bbox_inches='tight', format='png', transparent=True, dpi=150)

fig = plt.figure(3)
plt.clf()
ax1 = fig.add_subplot(111)
ax1.plot(burn_start_heights,burn_time,color=colors[0])
ax1.set_ylabel('Burn time $t_{\\mathrm{b}}$ [s]')
ax1.spines['right'].set_color(colors[1]) 
ax1.spines['left'].set_color(colors[0]) 
ax1.tick_params(axis='y', colors=colors[0])
ax1.yaxis.label.set_color(colors[0])
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
ax2.plot(burn_start_heights,np.array(burn_thrust)/1e3,color=colors[1])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors=colors[1])
ax2.yaxis.label.set_color(colors[1])
ax2.set_ylabel('Thrust $T$ [kN]')
ax1.set_xlabel('Burn start height $H_1$ [m]')
plt.autoscale(tight=True)
plt.tight_layout()
plt.show()
fig.savefig('media/burn_time.png',
            bbox_inches='tight', format='png', transparent=True, dpi=150)