"""
Finds the relationship between burn starting height, burn time, final mass and
thrust for the simple 1-d rocket landing problem without drag.

D. Malyuta -- ACL, University of Washington
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt

import progressbar # sudo -H pip install progressbar2

# Progress bar appearance
pb_widgets=[progressbar.Percentage(),' ',
            progressbar.Bar(),' (',progressbar.ETA(),')']

# LaTeX font in plots
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rc('font',**{'family':'serif'})
matplotlib.rc('text', usetex=True)

class Rocket:
    def __init__(self,noisy=False,controlled=False):
        """
        Defines parameters of the rocket landing problem.
        
        Parameters
        ----------
        noisy : bool
            If `True`, inject noise into the dynamics. This makes the open-loop
            application of the nominal landing problem solution fail to soft-
            land the rocket.
        """
        # Environment parameters
        self.g = 9.81 # [m/s^2] Acceleration due to gravity
        
        # Rocket parameters
        self.m_0 = 1905. # [kg] Initial mass
        self.H_0 = 300. # [m] Initial height
        self.v_0 = 100. # [m/s] Initial velocity
        self.I_sp = 225 # [s] Rocket engine specific impulse

        self.noisy = noisy
        if self.noisy:
            # Noise model parameters
            self.n_T = 2000. # [N] Thrust noise. T_actual\in[T_des-n_T,T_des+n_T]
            self.noise_freq = 20. # [Hz] Frequency of noise changing
            self.noise_t_last = -1/self.noise_freq
            
        self.controlled = controlled

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
        h_i,v_i,m_i,h,v,m = state[0],state[1],state[2],state[3],state[4],state[5]
        
        # Ideal system's dynamics
        dhidt = -v_i
        T_desired = self.thrust(h_i) # Determine rocket engine thrust
        dvidt = self.g-T_desired/m_i
        dmidt = -T_desired/(self.I_sp*self.g)

        # Control system
        T_adjust = 0.
        if self.controlled:
            v_error = v_i-v
#            u_tilde = 1.*v_error
#            alpha = 1/(self.I_sp*self.g)
#            T_adjust = m*(g-u_tilde/alpha)-T_desired
            proportional_gain = 2000.
            T_tilde = proportional_gain*v_error
            T_adjust = m*(g-T_tilde)-T_desired
        
        # Real system's dynamics
        dhdt = -v
        if self.noisy and time >= self.noise_t_last+1/self.noise_freq:
            if T_desired > 0:
                self.T_error = np.random.uniform(-self.n_T,self.n_T)
            else:
                # No "leaking thrust" when engine is not turned on, i.e. prior
                # to burn
                self.T_error = 0.
            self.noise_t_last = time
        elif not self.noisy and time==0:
            self.T_error = 0.
        T_actual = T_desired+T_adjust+self.T_error
        dvdt = self.g-T_actual/m
        dmdt = -T_actual/(self.I_sp*self.g)
        dxdt = np.array([dhidt,dvidt,dmidt,dhdt,dvdt,dmdt])
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
        v = state[4]
        return v
    ascentEvent.direction = -1. # Event triggers when starting to go up
    ascentEvent.terminal = True # Integration will stop once going up
    
    def burnEvent(self,time,state):
        """
        Indicates the start of the burn event.
        """
        h = state[3]
        return h-self.H_1
    burnEvent.direction = -1. # Event triggers when going below H_1

def simulate(rocket):
    """
    Simulate the rocket landing problem for a particular burn start height H_1.
    
    Parameters
    ----------
    H_1 : float
        Burn start height.
        
    Returns
    -------
    TODO
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

trial_count = 50
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
ax.set_xlabel('Height at hover [m]')
ax.set_ylabel('Bin count')
ax.legend()
plt.show()