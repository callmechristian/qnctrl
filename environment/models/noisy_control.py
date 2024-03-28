from ..core import ctrlPolar, entangler, noisyStat
from ..random_motion import LadyBug

import numpy as np


class SimpleControlledEnv:    
    def __init__(self, t0: float = 0, max_t: float = 0.2, latency: int = 3):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        # the polarization vector of the pump
        self.H = 1/np.sqrt(2)*np.matrix([[1],[0]])

        self.phi = []
        for i in range (12):
            self.phi.append(LadyBug())

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
        
        self.ctrl_pump = np.zeros(4)
        """
        The pump control array.

        This variable represents the control array for the pump. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_alice = np.zeros(4)
        """
        The Alice control array.

        This variable represents the control array for Alice. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_bob = np.zeros(4)
        """
        The Bob control array.

        This variable represents the control array for Bob. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.latency = latency
        """
        The control latency.

        This variable represents the number of steps the control is delayed. It also represents the number of steps 
        included in the MDP state.
        """
        
        self.done = False
        
        self.QBER_history = []
        self.phi_history = []

    def step(self, a_pump: np.array = np.zeros(4), a_alice: np.array = np.zeros(4), a_bob: np.array = np.zeros(4)):
        # set self control gates to action
        self.ctrl_pump = a_pump
        self.ctrl_alice = a_alice
        self.ctrl_bob = a_bob
                
        # *: assume our MDP state is the size of the latency in control
        for ctrl_latency_counter in range(self.latency + 1):
            # update current time step
            self.t += self.delta_t

            # compute the move the angles based on the motion model
            phi_move = []
            for i in range(12):
                phi_move.append(self.phi[i].move(self.t))

            # rotation of the pump in the source -- + 
            # ?: here is where we do the control with @gate
            pumpPolarisation = ctrlPolar(phi_move[0:4]) @ self.H
            if ctrl_latency_counter == self.latency:
                pumpPolarisation = ctrlPolar(self.ctrl_pump) @ pumpPolarisation
            
            # generation of the entangled state
            entangledState = entangler(pumpPolarisation)
            # rotation of the entangled state during the propagation -- gives entangled state at next time step
            entangledStatePropag = np.kron(ctrlPolar(phi_move[4:8]),
                                        ctrlPolar(phi_move[8:12])) @ entangledState
            
            # ?: here is where we do the control with np.kron
            if ctrl_latency_counter == self.latency:
                entangledStatePropag = np.kron(ctrlPolar(self.ctrl_alice), ctrlPolar(self.ctrl_bob)) @ entangledStatePropag
            
            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            QBERs_current = noisyStat(entangledStatePropag)
            self.QBER_history.append(QBERs_current)
            
            # if we exceed max t
            if self.t >= self.max_t:
                self.done = True
                break
        
        return self.get_state(), self.get_reward(), self.get_done()
    
    def reset(self):
        self.t = 0.
        self.done = False
        self.QBER_history = []
        self.phi_history = []
        self.step()
        return self.get_state()
    
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
        
    def get_state(self):
        """
        Returns the current state of the environment as a numpy array of the two QBERs.

        Returns:
            np.array(2): first QBERz, then QBERx
        """
        return self.QBER_history[-1] 
    
    def get_reward(self):
        QBER = self.QBER_history[-1]  # assuming this is where you store your QBERs
        reward = -1 * (QBER[0] + QBER[1])
        return reward

    def get_done(self):
        return self.t >= self.max_t
    
    def get_info(self):
        return self.t