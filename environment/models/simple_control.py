"""
This module provides a class for simulating a simple controlled environment.

The SimpleControlledEnv class provides a simulation environment for a simple controlled system.
It allows for controlling the pump, Alice, and Bob, and computes the QBER
(Quantum Bit Error Rate) based on the control inputs.

The class uses the LadyBug class from the random_motion module to simulate random motion.

The module uses numpy for numerical operations and typing for type annotations.

Classes:
    SimpleControlledEnv: A class for simulating a simple controlled environment.
"""
from typing import List
import numpy as np

from ..core import polar_control, entangler, compute_qber
from ..random_motion import LadyBug

class SimpleControlledEnv:
    """
    A class representing a simple controlled environment.

    This class provides a simulation environment for a simple controlled system.
    It allows for controlling the pump, Alice, and Bob, and computes the QBER
    (Quantum Bit Error Rate) based on the control inputs.

    Attributes:
        t (float): The current time value.
        max_t (float): The maximum simulation time horizon.
        delta_t (float): The speed of the error fluctuation.
        ctrl_pump (np.array): The pump control array.
        ctrl_alice (np.array): The Alice control array.
        ctrl_bob (np.array): The Bob control array.
        latency (int): The control latency.
        ctrl_alice_current (np.array): The current Alice control array.
        ctrl_bob_current (np.array): The current Bob control array.
        ctrl_pump_current (np.array): The current pump control array.
        done (bool): Flag indicating if the simulation is done.
        qber_history (List[float]): The history of QBER values.
        phi_history (List[np.array]): The history of phi values.

    Methods:
        __init__(self, t0: float = 0, max_t: float = 0.2, latency: int = 3):
            Initializes an instance of SimpleControlledEnv.
        step(self, a_pump: np.array = np.zeros(4), a_alice: np.array = np.zeros(4),
             a_bob: np.array = np.zeros(4)):
            Performs a single step in the environment.
        reset(self):
            Resets the environment to its initial state.
        get_qber(self):
            Returns the history of QBER values.
        get_phi(self):
            Returns the history of phi values.
        get_state(self):
            Returns the current state of the environment.
        get_reward(self):
            Calculates and returns the reward based on the QBER history.
        get_done(self):
            Checks if the simulation is done.
        get_info(self):
            Returns the value of the 't' attribute.
    """

    def __init__(self, t0: float = 0, max_t: float = 0.2, latency: int = 3):
        """
        Initializes an instance of SimpleControlledEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.
            latency (int): The control latency. Default is 3.

        Returns:
            None
        """
        # the polarization vector of the pump
        self.H = 1/np.sqrt(2)*np.matrix([[1],[1]]) # pylint: disable=invalid-name

        self.phi = []
        for _ in range (12):
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

        This variable represents the control array for the
        pump. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_alice = np.zeros(4)
        """
        The Alice control array.

        This variable represents the control array for
        Alice. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_bob = np.zeros(4)
        """
        The Bob control array.

        This variable represents the control array for Bob. e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.latency = latency
        """
        The control latency.

        This variable represents the number of steps the control is delayed. It also represents
        the number of steps included in the MDP state.
        """

        self.ctrl_alice_current = np.zeros(4)
        self.ctrl_bob_current = np.zeros(4)
        self.ctrl_pump_current = np.zeros(4)


        self.done = False

        self.qber_history: List[float] = []

        self.phi_history: List[np.array] = []

    def step(self, a_pump: np.array = np.zeros(4), a_alice: np.array = np.zeros(4),
             a_bob: np.array = np.zeros(4)):
        """
        Perform a single step in the environment.

        Args:
            a_pump (np.array, optional): Control input for the pump. Defaults to np.zeros(4).
            a_alice (np.array, optional): Control input for Alice. Defaults to np.zeros(4).
            a_bob (np.array, optional): Control input for Bob. Defaults to np.zeros(4).

        Returns:
            tuple: A tuple containing the current state, reward, and done flag.
        """
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
            # *: here is where we do the control with @gate
            pump_polarisation = polar_control(phi_move[0:4]) @ self.H
            pump_polarisation = polar_control(self.ctrl_pump_current) @ pump_polarisation

            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation --
            # gives entangled state at next time step
            entangled_state_propagation = np.kron(polar_control(phi_move[4:8]),
                                        polar_control(phi_move[8:12])) @ entangled_state

            # *: here is where we do the control with np.kron
            entangled_state_propagation = np.kron(polar_control(self.ctrl_alice_current),
            polar_control(self.ctrl_bob_current)) @ entangled_state_propagation
            # *: update control actual values to the current control values
            if ctrl_latency_counter == self.latency:
                self.ctrl_alice_current = self.ctrl_alice
                self.ctrl_bob_current = self.ctrl_bob
                self.ctrl_pump_current = self.ctrl_pump

            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            qbers_current = compute_qber(entangled_state_propagation)
            self.qber_history.append(qbers_current)

            # if we exceed max t
            if self.t >= self.max_t:
                self.done = True
                break

        return self.get_state(), self.get_reward(), self.get_done()

    def reset(self):
        """
        Resets the environment to its initial state.
        
        This method resets the time `t` to 0, sets the `done` flag to False,
        clears the `qber_history` and `phi_history` lists, and calls the `step`
        method to perform an initial step. Finally, it returns the current state
        of the environment.
        
        Returns:
            state (object): The current state of the environment.
        """
        self.t = 0.
        self.done = False
        self.qber_history = []
        self.phi_history = []
        self.step()
        return self.get_state()

    def get_qber(self):
        """
        Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.
        
        Returns:
            numpy.ndarray: The history of QBER values.
        """
        return np.array(self.qber_history)

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
        return self.qber_history[-1]

    def get_reward(self):
        """
        Calculates and returns the reward based on the QBER history.

        Returns:
            float: The reward value.
        """
        qber = self.qber_history[-1]  # assuming this is where you store your QBERs
        reward = -1 * (qber[0] + qber[1])
        return reward

    def get_done(self):
        """
        Check if the current time step is greater than or equal to the maximum time step.

        Returns:
            bool: True if the current time step is greater than or equal to the maximum
            time step, False otherwise.
        """
        return self.t >= self.max_t

    def get_info(self):
        """
        Returns the value of the 't' attribute.
        
        Returns:
            The value of the 't' attribute.
        """
        return self.t
