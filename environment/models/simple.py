from ..core import ctrlPolar, entangler, QBERs
from ..random_motion import ladybug

import numpy as np


class SimpleEnv:
    def __init__(self, t0: float = 0, max_t: float = 0.2):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        self.H = 1/np.sqrt(2)*np.matrix([[1],[0]])

        self.phi = []
        for i in range (12):
            self.phi.append(ladybug())

        self.t = t0 + 0.
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_t + self.t
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
    
    def simulate(self):
        for t in np.arange(0., self.max_t, self.delta_t):
            # move the angles based on the motion model
            phi_move = []
            for i in range(12):
                phi_move.append(self.phi[i].move(t))
                
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
