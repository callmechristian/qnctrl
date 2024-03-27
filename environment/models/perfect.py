from ..core import ctrlPolar, entangler, QBERs

import numpy as np


class PerfectEnv:
    def __init__(self, t0: float = 0, max_t: float = 0.2):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        self.H = 1/np.sqrt(2)*np.matrix([[1],[1]])

        self.phi = []
        for i in range (12):
            self.phi.append(0)

        self.t = t0 + 0.
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_t # ! will be updated when simulate is called
        self.initial_max_t = max_t
        """
        The maximum simulation time horizon.

        This variable represents the maximum simulation time.
        """
        self.delta_t = 0.0001 # speed of the error fluctuation
        """
        The speed of the error fluctuation.

        The frequency of the error model: lower values mean less fluctuation.
        """
        
        self.QBER_history = []
        self.phi_history = []
        self.t_history = []
    
    def simulate(self, reset=True):
        if reset:
            self.reset()
        
        for t in np.arange(self.t, self.max_t + self.t, self.delta_t):
            # dont move the angles
            phi_move = np.zeros(12)
                
            # rotation of the pump in the source -- + 
            # TODO: here is where we do the control with @gate
            pumpPolarisation = ctrlPolar(phi_move[0:4])@self.H
            # generation of the entangled state
            entangledState = entangler(pumpPolarisation)
            # rotation of the entangled state during the propagation -- gives entangled state at next time point
            entangledStatePropag = np.kron(ctrlPolar(phi_move[4:8]),
                                        ctrlPolar(phi_move[8:12]))@entangledState
            # TODO: here is where we do the control with np.kron
            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            self.QBER_history.append(QBERs(entangledStatePropag))
            # append time for plotting
            self.t_history.append(t)
        
        # update times
        self.t = self.t + self.max_t
        self.max_t = self.max_t + self.initial_max_t
    
    def simulate_no_polar(self, reset=True):
        if reset:
            self.reset()
        print("what")
        for t in np.arange(self.t, self.max_t + self.t, self.delta_t):                
            # rotation of the pump in the source -- + 
            pumpPolarisation = self.H
            # generation of the entangled state
            entangledState = entangler(pumpPolarisation)
            # rotation of the entangled state during the propagation -- gives entangled state at next time point
            entangledStatePropag = entangledState
            # compute the QBERs
            self.QBER_history.append(QBERs(entangledStatePropag))
            # append time for plotting
            self.t_history.append(t)
        
        # update times
        self.t = self.t + self.max_t
        self.max_t = self.max_t + self.initial_max_t
    
    def get_QBER(self):
            """
            Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.
            
            Returns:
                numpy.ndarray: The history of QBER values.
            """
            return np.array(self.QBER_history)
    
    def get_phi(self):
            """
            Returns the phi history as a numpy array.

            Returns:
                numpy.ndarray: The phi history.
            """
            return np.array(self.phi_history)
        
    def reset(self):
        self.t = 0.
        self.t_history = []
        self.QBER_history = []
        self.phi_history = []
