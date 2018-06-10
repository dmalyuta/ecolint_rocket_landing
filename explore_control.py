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

def simulate(rocket):
    """
    Simulate the rocket landing problem for a particular burn start height H_1.
    
    Parameters
    ----------
    H_1 : float
        Burn start height.
        
    Returns
    -------
    final_height : float
        The height at which the rocket began ascending (i.e. the hover height).
    """
    rocket.setLandingBurnStartHeight(150.)
    
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
    h = state.y[3]
    
    final_height = h[-1]
    
    return final_height

trial_count = 1000
final_height_without_control, final_height_with_control = [], []
for _ in progressbar.progressbar(range(trial_count), widgets=pb_widgets):
    rocket_without_control = Rocket(noisy=True,controlled=False)
    final_height_without_control.append(simulate(rocket_without_control))
    rocket_with_control = Rocket(noisy=True,controlled=True)
    final_height_with_control.append(simulate(rocket_with_control))

#%% Plot result
    
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.hist([final_height_without_control,final_height_with_control],
        label=['Uncontrolled','Controlled'])
ax.set_xlabel('Height at hover $H_1$ [m]')
ax.set_ylabel('Bin count')
ax.legend()
plt.show()
fig.savefig('media/control_effect_histogram.png',
            bbox_inches='tight', format='png', transparent=True, dpi=150)