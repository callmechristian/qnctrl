from ..core import ctrlPolar, entangler, QBERs
from ..random_motion import ladybug

import numpy as np


class SimpleControlledFixedEnv:    
    def __init__(self, t0: float = 0, max_t: float = 0.2, latency: int = 3, fixed_error: np.array = np.zeros(12)):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        # the polarization vector of the pump
        self.H = 1/np.sqrt(2)*np.matrix([[1],[1]])

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
        self.fixed_error_ctrl_pump = fixed_error[0:4]
        """
        The simulated error (array) for the pump.

        This variable represents the simulated error for the pump. It is used to simulate the error for the pump entanglement propagation.
        """
        self.fixed_error_ctrl_alice = fixed_error[4:8]
        """
        The simulated error (array) for Alice.

        This variable represents the simulated error for Alice. It is used to simulate the error for Alice's entanglement propagation.
        """
        self.fixed_error_ctrl_bob = fixed_error[8:12]
        """
        The simulated error (array) for Bob.

        This variable represents the simulated error for Bob. It is used to simulate the error for Bob's entanglement propagation.
        """
        self.fixed_errors_flags = np.repeat(False, 12)
        """
        The fixed error flags.
        
        This variable represents the flags for whcih errors will be fixed.
        """
        self.ctrl_alice_current = np.zeros(4)
        self.ctrl_bob_current = np.zeros(4)
        self.ctrl_pump_current = np.zeros(4)
        
        self.done = False
        """
        The done flag.
        
        This variable represents the flag that indicates the end of the simulation.
        """
        
        self.QBER_history = []
        """
        The QBER history.
        
        This variable represents the history of the QBER values.
        """
        self.phi_history = []
        """
        The phi history.
        
        This variable represents the history of the phi values.
        """

    def step(self, a_pump: np.array = np.zeros(4), a_alice: np.array = np.zeros(4), a_bob: np.array = np.zeros(4)):
        # set self control gates to action
        self.ctrl_pump = a_pump
        self.ctrl_alice = a_alice
        self.ctrl_bob = a_bob
                
        # *: assume our MDP state is the size of the latency in control
        for ctrl_latency_counter in range(self.latency + 1):
            # update current time step
            self.t += self.delta_t

            # compute the move the angles based on the motion model or fixed
            phi_move = []
            # concatenate the fixed errors
            _errs = np.concatenate((self.fixed_error_ctrl_alice, self.fixed_error_ctrl_bob, self.fixed_error_ctrl_pump))
            for i in range(12):
                # if the error is fixed, we append the fixed error
                if self.fixed_errors_flags[i]:
                    phi_move.append(_errs[i])
                else:
                    # otherwise we append the random error
                    phi_move.append(self.phi[i].move(self.t))

            # rotation of the pump in the source -- + 
            # *: here is where we do the control with @gate
            pumpPolarisation = ctrlPolar(phi_move[0:4]) @ self.H
            pumpPolarisation = ctrlPolar(self.ctrl_pump_current) @ pumpPolarisation
            
            # generation of the entangled state
            entangledState = entangler(pumpPolarisation)
            # rotation of the entangled state during the propagation -- gives entangled state at next time step
            entangledStatePropag = np.kron(ctrlPolar(phi_move[4:8]),
                                        ctrlPolar(phi_move[8:12])) @ entangledState
            
            # *: here is where we do the control with np.kron
            entangledStatePropag = np.kron(ctrlPolar(self.ctrl_alice_current), ctrlPolar(self.ctrl_bob_current)) @ entangledStatePropag
            # *: update control actual values to the current control values
            if ctrl_latency_counter == self.latency:
                self.ctrl_alice_current = self.ctrl_alice
                self.ctrl_bob_current = self.ctrl_bob
                self.ctrl_pump_current = self.ctrl_pump
            
            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            QBERs_current = QBERs(entangledStatePropag)
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