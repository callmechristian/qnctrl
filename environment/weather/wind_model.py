"""
This module defines the WindModel class and associated functions for simulating
the effects of wind on a fibre cable.

Classes:
- WindModel: Represents a wind model that computes samples for a given fibre cable.

Functions:
- area_hit(wind_angle: float, fibre_cable: FibreLink) -> float: Calculates the
area of the fibre cable that is hit by the wind.

Constants:
- AIR_DENSITY: The density of air at sea level, used in wind force calculations.
"""

import numpy as np

from ..core.fibre_cable import FibreLink
from ..core.polarisation_disturbance import polarisation_from_force

# density of air
AIR_DENSITY = 1.229  # kg/m^3 -- at sea level


class WindModel:
    """
    Represents a wind model that computes samples for a given fibre cable.

    Attributes:
    - index: The current index of the wind model.
    - wind_speed: A list of wind speeds.
    - wind_direction: A list of wind directions.

    Methods:
    - __init__(self, data): Initializes the WindModel object.
    - reset(self): Resets the index of the wind model to 0.
    - compute_sample(self, fibre_cable): Computes the sample for the given fibre cable.
    - next_sample(self): Gets the next sample from the wind model.
    """

    def __init__(self, data):
        self.index = 0
        self.wind_speed = data["wind_speed"]
        self.wind_direction = data["wind_direction"]

    def reset(self):
        """
        Resets the index of the wind model to 0.
        """
        self.index = 0

    def compute_sample(self, fibre_cable: FibreLink):
        """
        Computes the sample for the given fibre cable.

        Parameters:
        - fibre_cable: The FibreLink object representing the fibre cable.

        Returns:
        - A tuple containing the polarisation vector and the force applied to
        the fibre cable. (np.array, float)
        """

        # pylint: disable=invalid-name
        F = (
            1
            / 2
            * AIR_DENSITY
            * area_hit(self.wind_direction[self.index], fibre_cable)
            * self.wind_speed[self.index] ** 2
            * fibre_cable.drag_coefficient
        ) # N
        # pylint: enable=invalid-name

        self.next_sample()

        # moment of force will be according to incident wind
        angle_in_rad = self.wind_direction[self.index]
        alpha = np.pi / 2 - angle_in_rad
        force_rot_dir = np.matrix(
            [
                [np.cos(alpha), np.sin(alpha)],
                [-np.sin(alpha), np.cos(alpha)],
            ]
        )
        return (polarisation_from_force(F, fibre_cable) @ force_rot_dir, F)

    def next_sample(self):
        """
        Get the next sample from the wind model.

        Returns:
            The next sample from the wind model.
        """
        self.index += 1


def area_hit(wind_angle: float, fibre_cable: FibreLink):
    """
    Calculate the area of the fibre cable that is hit by the wind.

    Parameters:
    - wind_angle (float): The angle of the wind in degrees.
    - fibre_cable (FibreLink): An instance of the FibreLink class representing the fibre cable.

    Returns:
    - float: The area of the fibre cable that is hit by the wind in square meters.
    """
    return (
        2
        * np.pi
        * fibre_cable.cable_radius
        * fibre_cable.cable_length
        * np.cos(wind_angle * np.pi / 180)
    )  # m^2 #! assume 0 degrees means wind is perpendicular to the cable
