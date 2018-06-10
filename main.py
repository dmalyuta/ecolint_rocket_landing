"""
Finds the relationship between burn starting height, burn time, final mass and
thrust for the simple 1-d rocket landing problem without drag.

D. Malyuta -- ACL, University of Washington
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class Rocket:
    def __init__(self):
        """
        Defines parameters of the rocket landing problem.
        """
        # Environment parameters
        self.g = 9.81 # [m/s^2] Acceleration due to gravity
        
        # Rocket parameters
        self.m_0 = 1905. # [kg] Initial mass
        self.H_0 = 300. # [m] Initial height
        self.v_0 = 100. # [m/s] Initial velocity
        self.I_sp = 225 # [s] Rocket engine specific impulse
        
    def setLandingBurnStartHeight(self,H_1):
        """
        Sets the landing burn start height and internall computes the
        associated control quantities.
        
        Parameters
        ----------
        H_1 : float
            Landing burn start height.
        """
        self.H_1 = H_1
        
        #### Compute the landing burn properties
        v0,m0,Isp,g,H0,H1 = self.v_0,self.m_0,self.I_sp,self.g,self.H_0,H_1
        # Velocity at burn start [m/s]
        self.v_1 = np.sqrt(v0**2+2*g*(H0-H1))
        v1 = self.v_1
        # Find the thrust and burn time numerically by solving for the
        # necesarsy conditions they must satisfy
        alpha = 1/(Isp*g)
        def thrustBurnTimeNecessaryConditions(x):
            T,tb = x[0],x[1]
            cond1 = g*tb+1/alpha*np.log(1-alpha*T*tb/m0)+v1
            cond2 = H1+0.5*g*tb**2+tb/alpha+m0/(alpha**2*T)*np.log(1-alpha*T*tb/m0)
            cond = np.array([cond1,cond2])
            return cond
        # Initial guess from rocket dynamics with constant mass (i.e. double integrator)
        T_guess = m0*v1**2/(2*H1)+m0*g
        tb_guess = v1/(T_guess/m0-g)
        guess = np.array([T_guess,tb_guess])
        sol = fsolve(thrustBurnTimeNecessaryConditions, guess)
        # Burn thrust [N]
        self.T = sol[0]
        # Burn time [s]
        self.t_b = sol[1]
        # Final mass [s], according to Tsiolkovsky rocket equation
        self.m_f = m0*np.exp(-v1/(Isp*g))
        
    def thrust(self,height):
        """
        Controls the rocket engine for the landing burn.
        
        Parameters
        ----------
        height : float
            Rocket height.
            
        Returns
        -------
        : float
            Roket engine desired thrust.
        """
        return self.T if height<=self.H_1 else 0.
        
    def dynamics(self,time,state):
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
        T = self.thrust(h) # Determine rocket engine thrust
        dvdt = self.g-T/m
        dmdt = -T/(self.I_sp*self.g)
        dxdt = np.array([dhdt,dvdt,dmdt])
        return dxdt
    
    def ascentEvent(self,time,state):
        """
        Indicates the event of rocket has started to go up.
        
        Parameters
        ----------
        time : float
            Current time.
        state : array
            Current state [position,velocity,mass].
            
        Returns
        -------
        v : float
            Rocket velocity (positive down).
        """
        v = state[1]
        return v
    ascentEvent.direction = -1. # Event triggers when starting to go up
    ascentEvent.terminal = True # Integration will stop once going up
    
    def burnEvent(self,time,state):
        """
        Indicates the start of the burn event.
        """
        h = state[0]
        return h-self.H_1
    burnEvent.direction = -1. # Event triggers when going below H_1
        
rocket = Rocket()
rocket.setLandingBurnStartHeight(100.)

# Simulate the dynamics
t_f = 100.
x_0 = np.array([rocket.H_0,rocket.v_0,rocket.m_0])

state = solve_ivp(fun = rocket.dynamics,
                  t_span = (0,t_f),
                  y0 = x_0,
                  events = [rocket.burnEvent,rocket.ascentEvent],
                  max_step = 0.001)

# Extract simulation results
t = state.t
t_burn = state.t_events[0]
t_ascent = state.t_events[1]
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
