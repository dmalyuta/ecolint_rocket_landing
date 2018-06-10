"""
Finds the relationship between burn starting height, burn time, final mass and
thrust for the simple 1-d rocket landing problem without drag.

D. Malyuta -- ACL, University of Washington
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Rocket:
    def __init__(self):
        """
        Defines rocket parameters.
        """
        self.m_0 = 1905. # [kg] Initial mass
        self.H_0 = 300. # [m] Initial height
        self.v_0 = 100 # [m/s] Initial velocity
        self.g = 9.81 # [m/s^2] Acceleration due to gravity
        self.I_sp = 225 # [s] Rocket engine specific impulse
        
    def dynamics(self,time,state,thrust):
        """
        Returns the time derivative of the state=[position,velocity,mass].
        
        Parameters
        ----------
        time : float
            Current time.
        state : array
            Current state [position,velocity,mass].
        thrust : float
            Current thrust produced by the rocket engine (>=0).
            
        Returns
        -------
        dxdt : array
            Time derivative of the state.
        """
        h,v,m = state[0],state[1],state[2]
        dhdt = -v
        dvdt = self.g-thrust/m
        dmdt = -thrust/(self.I_sp*self.g)
        dxdt = np.array([dhdt,dvdt,dmdt])
        return dxdt
    
    def touchdown_event(self,time,state):
        """
        Indicates the event of rocket contact with the ground when the output
        goes from positive to negative.
        
        Parameters
        ----------
        time : float
            Current time.
        state : array
            Current state [position,velocity,mass].
            
        Returns
        -------
        h : float
            Height of rocket about ground
        """
        h = state[0]
        return h
    touchdown_event.direction = -1. # Event triggers when going below ground
    touchdown_event.terminal = True # Integration will stop at touchdown
        
rocket = Rocket()

# Simulate the dynamics
t_f = 100.
x_0 = np.array([rocket.H_0,rocket.v_0,rocket.m_0])

state = solve_ivp(fun = lambda t,x: rocket.dynamics(t,x,0),
                  t_span = (0,t_f),
                  y0 = x_0,
                  events = rocket.touchdown_event,
                  max_step = 0.02)

# Extract simulation results
t = state.t
t_touchdown = state.t_events[0]
h = state.y[0]
v = state.y[1]
m = state.y[2]

# Plot result
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(311)
ax.plot(t,h)
ax.set_ylabel('Height [m]')
plt.autoscale(tight=True)
ax = fig.add_subplot(312)
ax.plot(t,v)
ax.set_ylabel('Velocity [m/s]')
plt.autoscale(tight=True)
ax = fig.add_subplot(313)
ax.plot(t,m)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Mass [kg]')
plt.autoscale(tight=True)
plt.tight_layout()
plt.show()