"""
This module defines the PerfectEnv class, which represents a perfect quantum environment.

The PerfectEnv class provides methods to simulate a perfect quantum system over a given time period,
compute the Quantum Bit Error Rate (QBER), and retrieve the simulation history.

Attributes:
    H (numpy.matrix): The Hadamard gate matrix.
    phi (list): The list of angles for the control gates.
    t (float): The current time value.
    max_t (float): The maximum simulation time horizon.
    initial_max_t (float): The initial maximum simulation time horizon.
    delta_t (float): The speed of the error fluctuation.
    qber_history (list): The history of QBER values.
    phi_history (list): The history of phi values.
    t_history (list): The history of time values.
"""
from typing import List
import numpy as np

from ..core import polar_control, entangler, compute_qber

class PerfectEnv:
    """
    Represents a perfect quantum environment.

    This class provides methods to simulate a perfect quantum system over a given time period,
    compute the Quantum Bit Error Rate (QBER), and retrieve the simulation history.

    Attributes:
        H (numpy.matrix): The Hadamard gate matrix.
        phi (list): The list of angles for the control gates.
        t (float): The current time value.
        max_t (float): The maximum simulation time horizon.
        initial_max_t (float): The initial maximum simulation time horizon.
        delta_t (float): The speed of the error fluctuation.
        qber_history (list): The history of QBER values.
        phi_history (list): The history of phi values.
        t_history (list): The history of time values.
    """

    def __init__(self, t0: float = 0, max_t: float = 0.2):
        """
        Initializes an instance of PerfectEnv.

        Parameters:
            t0 (float): The initial time value. Default is 0.
            max_t (float): The maximum simulation time horizon. Default is 0.2.

        Returns:
            None
        """
        self.H = 1/np.sqrt(2)*np.matrix([[1],[1]]) # pylint: disable=invalid-name

        self.phi = []
        for _ in range (12):
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

        self.qber_history: List[float] = []
        self.phi_history: list = []
        self.t_history: list = []

    def simulate(self, reset=True):
        """
        Simulates the quantum system over a given time period.

        Args:
            reset (bool, optional): Indicates whether to reset the system before simulation. 
                Defaults to True.

        Returns:
            None
        """
        if reset:
            self.reset()

        for t in np.arange(self.t, self.max_t + self.t, self.delta_t):
            # dont move the angles
            phi_move = np.zeros(12)

            # rotation of the pump in the source -- +
            # //: here is where we do the control with @gate
            pump_polarisation = polar_control(phi_move[0:4])@self.H
            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation -- gives
            # entangled state at next time point
            entangled_state_propagation = np.kron(polar_control(phi_move[4:8]),
                                        polar_control(phi_move[8:12]))@entangled_state
            # //: here is where we do the control with np.kron
            # append the angles for plotting
            self.phi_history.append(phi_move)
            # compute the QBERs
            self.qber_history.append(compute_qber(entangled_state_propagation))
            # append time for plotting
            self.t_history.append(t)

        # update times
        self.t = self.t + self.max_t
        self.max_t = self.max_t + self.initial_max_t

    def simulate_no_polar(self, reset=True):
        """
        Simulates the system without considering polarization effects.

        Args:
            reset (bool, optional): Indicates whether to reset the system before
            simulation. Defaults to True.
        """
        if reset:
            self.reset()
        print("what")
        for t in np.arange(self.t, self.max_t + self.t, self.delta_t):
            # rotation of the pump in the source -- +
            pump_polarisation = self.H
            # generation of the entangled state
            entangled_state = entangler(pump_polarisation)
            # rotation of the entangled state during the propagation -- gives
            # entangled state at next time point
            entangled_state_propagation = entangled_state
            # compute the QBERs
            self.qber_history.append(compute_qber(entangled_state_propagation))
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
        return np.array(self.qber_history)

    def get_phi(self):
        """
        Returns the phi history as a numpy array.

        Returns:
            numpy.ndarray: The phi history.
        """
        return np.array(self.phi_history)

    def reset(self):
        """
        Resets the state of the object.

        This method sets the value of `t` to 0 and clears the history lists `t_history`,
        `QBER_history`, and `phi_history`.

        Parameters:
            None

        Returns:
            None
        """
        self.t = 0.
        self.t_history = []
        self.qber_history = []
        self.phi_history = []
