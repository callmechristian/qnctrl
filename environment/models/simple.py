"""
This module provides a class for simulating a simple quantum control environment.

The SimpleEnv class provides methods to simulate the behavior of a quantum control system.
It allows for controlling the pump and the entangled state, and computes the QBER
(Quantum Bit Error Rate) based on the control inputs.

The class uses the LadyBug class from the random_motion module to simulate random motion.

The module uses numpy for numerical operations and typing for type annotations.

Classes:
    SimpleEnv: A class for simulating a simple quantum control environment.
"""

from typing import List
import numpy as np

from ..core import polar_control, entangler, compute_qber
from ..random_motion import LadyBug


class SimpleEnv:
    """
    A class representing a simple environment for quantum control simulations.

    This class provides methods to simulate the behavior of a quantum control system.
    """

    def __init__(self, t0: float = 0, max_t: float = 0.2):
        """
        Initializes an instance of SimpleEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        self.H = 1 / np.sqrt(2) * np.matrix([[1], [1]])  # pylint: disable=invalid-name

        self.phi = []
        for _ in range(12):
            self.phi.append(LadyBug())

        self.t = t0 + 0.0
        """
        The initial time value.

        This variable represents the starting time for the simulation.
        """
        self.max_t = max_t  # ! will be updated when simulate is called
        self.initial_max_t = max_t
        """
        The maximum simulation time horizon.

        This variable represents the maximum simulation time.
        """
        self.delta_t = 0.0001  # speed of the error fluctuation
        """
        The speed of the error fluctuation.

        The frequency of the error model: lower values mean less fluctuation.
        """

        self.qbers_history: List[float] = []
        self.phi_history: List[np.array] = []  # type: ignore
        self.t_history: List[float] = []

    def simulate(self, reset=True):
        """
        Simulates the behavior of the quantum control system.

        Parameters:
            reset (bool): Whether to reset the simulation. Default is True.

        Returns:
            None
        """
        if reset:
            self.reset()

        for t in np.arange(self.t, self.max_t + self.t, self.delta_t):
            # move the angles based on the motion model
            phi_move = []
            for i in range(12):
                phi_move.append(self.phi[i].move(t))

            # rotation of the pump in the source -- +
            # //: here is where we do the control with @gate
            pump_polarisation = polar_control(phi_move[0:4]) @ self.H
            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation --
            # gives entangled state at next time point
            entangled_state_propagation = (
                np.kron(polar_control(phi_move[4:8]), polar_control(phi_move[8:12]))
                @ entangled_state
            )
            # //: here is where we do the control with np.kron
            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            self.qbers_history.append(compute_qber(entangled_state_propagation))
            # append time for plotting
            self.t_history.append(t)

        # update times
        self.t = self.t + self.max_t
        self.max_t = self.max_t + self.initial_max_t

    def get_qber(self):
        """
        Returns the history of QBER (Quantum Bit Error Rate) as a numpy array.

        Returns:
            numpy.ndarray: The history of QBER values.
        """
        return np.array(self.qbers_history)

    def get_phi(self):
        """
        Returns the phi history as a numpy array.

        Returns:
            numpy.ndarray: The phi history.
        """
        return np.array(self.phi_history)

    def reset(self):
        """
        Resets the simulation.

        Returns:
            None
        """
        self.t = 0.0
        self.t_history = []
        self.qbers_history = []
        self.phi_history = []
