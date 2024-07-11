"""
This module provides a class for simulating the random motion of a ladybug.

The LadyBug class simulates the random motion of a ladybug based on a set of parameters 
that are randomly generated using a Gaussian distribution. The motion is calculated 
based on the current time and these parameters.

The module uses the random and numpy libraries for generating random numbers and 
performing mathematical operations respectively.

Classes:
    LadyBug: A class for simulating the random motion of a ladybug.
"""

from random import gauss, seed
from numpy import sin, cos


class LadyBug:
    """
    Represents a ladybug random motion model object.

    Attributes:
        params_x (list): A list of 8 parameters used for calculating the ladybug's movement.

    Methods:
        __init__(self, s: int = 0): Initializes a ladybug object with optional seed value.
        move(self, t): Calculates ladybug movement at time t.
    """

    def __init__(self, s: int = 0):
        """
        Initializes a ladybug model object.

        Args:
            s (int): Seed value for random number generation. Default is 0.
        """
        if s > 0:
            seed(s)
        
        self.params_amplitude = [gauss(0.0, 0.01) for u in range(4)]
        self.params_pulsation = [gauss(0.01, 0.1) for u in range(4)]

    def move(self, t: float):
        """
        Calculates the ladybug model's movement at time t.

        Args:
            t (float): Time value.

        Returns:
            float: The calculated movement of the ladybug model at time t.
        """
        [ax1, ax2, ax3, ax4] = self.params_amplitude
        [kx1, kx2, kx3, kx4] = self.params_pulsation

        # parameters close to a day representation
        ax1 = 1
        kx1 = 0.001

        x = (
            ax1 * sin(t * (kx1))
            + ax2 * cos(t * (kx2))
            + ax3 * sin(t * (kx3))
            + ax4 * cos(t * (kx4))
        )

        return x
