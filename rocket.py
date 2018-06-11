"""
Rocket landing problem class.

D. Malyuta -- ACL, University of Washington
"""

import numpy as np
from scipy.optimize import fsolve

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
        self.m_0 = 26e3 # [kg] Initial mass
        self.H_0 = 25e3 # [m] Initial height
        self.v_0 = 800. # [m/s] Initial velocity
        self.I_sp = 282. # [s] Rocket engine specific impulse
        self.T_1eng_max = 845e3 # [N] Maximum thrust of one rocket engine (Merlin 1D)

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
        # Initial guess from rocket dynamics with constant mass (i.e. double
        # integrator)
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
    
    def controller(self,state,T_nominal):
        """
        Computes an adjustment thrust to T_desired in order to bring the rocket
        to a softer touchdown despite uncertainty.
        
        Parameters
        ----------
        state : array
            Rocket dynamics state vector.
        T_nominal : float
            Nominal thrust, if the rocket dynamics were perfect (no noise).
            
        Returns
        -------
        T_adjust : float
            Adjustment to the thrust amount.
        """
        v_i,v,m = state[1],state[4],state[5]
        
        T_adjust = 0.
        if self.controlled:
            v_error = v_i-v
            proportional_gain = 5.
            T_tilde = proportional_gain*v_error
            T_adjust = -m*T_tilde
            
        return T_adjust
        
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
        h_i,v_i,m_i,v,m = state[0],state[1],state[2],state[4],state[5]
        
        # Ideal system's dynamics
        dhidt = -v_i
        T_nominal = self.thrust(h_i) # Determine rocket engine thrust
        dvidt = self.g-T_nominal/m_i
        dmidt = -T_nominal/(self.I_sp*self.g)

        # Control system
        T_adjust = self.controller(state,T_nominal)
        
        # Real system's dynamics
        dhdt = -v

        # Determine thrust noise component        
        if self.noisy and time >= self.noise_t_last+1/self.noise_freq:
            if T_nominal > 0:
                self.T_error = np.random.uniform(-self.n_T,self.n_T)
            else:
                # No "leaking thrust" when engine is not turned on, i.e. prior
                # to burn
                self.T_error = 0.
            self.noise_t_last = time
        elif not self.noisy and time==0:
            self.T_error = 0.
        T_actual = T_nominal+T_adjust+self.T_error
        
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