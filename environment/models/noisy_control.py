"""
This module contains the implementation of the SimpleControlledEnv class,
which simulates a quantum communication environment with a controlled
quantum bit error rate (QBER).

The SimpleControlledEnv class models a quantum communication system with
a pump and two users, Alice and Bob. The pump generates an entangled state,
which is then propagated to Alice and Bob. The state of the system is represented
by the QBER, which measures the error rate of the quantum communication.

The class provides methods to initialize the environment, step through the
simulation with specified control actions for the pump, Alice, and Bob, and
reset the environment to its initial state. It also provides methods to
retrieve the current state, reward, and other information about the environment.

The module also imports necessary functions and classes from other modules,
such as polar_control, entangler, compute_noisy_qber from the core module,
and LadyBug from the random_motion module.
"""

from typing import List
import numpy as np

from ..core import polar_control, entangler, compute_noisy_qber
from ..random_motion import LadyBug


class SimpleControlledEnv:
    """
    A simple controlled environment for quantum control simulations.

    This class represents an environment for simulating quantum control experiments.
    It allows for controlling the pump, Alice, and Bob operations to manipulate the
    entangled state and compute the Quantum Bit Error Rate (QBER).

    Attributes:
        t (float): The current time value.
        max_t (float): The maximum simulation time horizon.
        delta_t (float): The speed of the error fluctuation.
        ctrl_pump (np.array): The pump control array.
        ctrl_alice (np.array): The Alice control array.
        ctrl_bob (np.array): The Bob control array.
        latency (int): The control latency.
        done (bool): Indicates if the simulation is done.
        QBER_history (list): The history of QBER values.
        phi_history (list): The history of phi values.

    Methods:
        __init__(self, t0=0, max_t=0.2, latency=3):
            Initializes an instance of SimpleControlledEnv.

        step(self, a_pump=np.zeros(4), a_alice=np.zeros(4), a_bob=np.zeros(4)):
            Performs a simulation step.

        reset(self):
            Resets the environment to its initial state.

        get_qber(self):
            Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.

        get_phi(self):
            Returns the phi history as a numpy array.

        get_state(self):
            Returns the current state of the environment as a numpy array of the two QBERs.

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
        self.H = 1 / np.sqrt(2) * np.matrix([[1], [0]]) # pylint: disable=invalid-name

        self.phi = []
        for _ in range(12):
            self.phi.append(LadyBug())

        self.t = t0 + 0.0
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_t + self.t
        """
        The maximum simulation time horizon.

        This variable represents the maximum simulation time.
        """
        self.delta_t = 0.0001  # speed of the error fluctuation
        """
        The speed of the error fluctuation.

        The frequency of the error model: lower values mean less fluctuation.
        """

        self.ctrl_pump = np.zeros(4)
        """
        The pump control array.

        This variable represents the control array for the pump.
        e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_alice = np.zeros(4)
        """
        The Alice control array.

        This variable represents the control array for Alice.
        e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.ctrl_bob = np.zeros(4)
        """
        The Bob control array.

        This variable represents the control array for Bob.
        e.g. np.array([0, 0, 0, 0]) for identity
        """
        self.latency = latency
        """
        The control latency.

        This variable represents the number of steps the control is delayed.
        It also represents the number of steps included in the MDP state.
        """

        self.done = False

        self.qber_history: List[float] = []
        self.phi_history: List[np.array] = [] # type: ignore

    def step(
        self,
        a_pump: np.array = np.zeros(4), # type: ignore
        a_alice: np.array = np.zeros(4), # type: ignore
        a_bob: np.array = np.zeros(4), # type: ignore
    ):
        """
        Performs a simulation step.

        This method updates the environment state based on the given control actions
        for the pump, Alice, and Bob. It computes the QBER, updates the time step,
        and checks if the simulation is done.

        Parameters:
            a_pump (np.array): The control array for the pump. Default is np.zeros(4).
            a_alice (np.array): The control array for Alice. Default is np.zeros(4).
            a_bob (np.array): The control array for Bob. Default is np.zeros(4).

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
                phi_move.append(self.phi[i].sample(self.t))

            # rotation of the pump in the source -- +
            # ?: here is where we do the control with @gate
            pump_polarisation = polar_control(phi_move[0:4]) @ self.H
            if ctrl_latency_counter == self.latency:
                pump_polarisation = polar_control(self.ctrl_pump) @ pump_polarisation

            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation --
            # gives entangled state at next time step
            entangled_state_propagation = (
                np.kron(polar_control(phi_move[4:8]), polar_control(phi_move[8:12]))
                @ entangled_state
            )

            # ?: here is where we do the control with np.kron
            if ctrl_latency_counter == self.latency:
                entangled_state_propagation = (
                    np.kron(
                        polar_control(self.ctrl_alice), polar_control(self.ctrl_bob)
                    )
                    @ entangled_state_propagation
                )

            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            qbers_current = compute_noisy_qber(entangled_state_propagation)
            self.qber_history.append(qbers_current)

            # if we exceed max t
            if self.t >= self.max_t:
                self.done = True
                break

        return self.get_state(), self.get_reward(), self.get_done()

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            state (object): The initial state of the environment.
        """
        self.t = 0.0
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
        qber = self.qber_history[-1]
        reward = -1 * (qber[0] + qber[1])
        return reward

    def get_done(self):
        """
        Checks if the current time step is greater than or equal to the maximum time step.

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
