"""
This module provides functionality to calculate the polarisation rotation
matrix based on the applied force and fibre cable properties.

Functions:
- polarisation_from_force(force: float, cable: FibreLink) -> numpy.matrix:
Calculates the polarisation rotation matrix based on the applied force and
cable properties.
"""

import numpy as np
from .fibre_cable import FibreLink


def polarisation_from_force(force: float, cable: FibreLink):
    """
    Calculates the polarisation rotation matrix based on the applied force and cable properties.

    Args:
        force (float): The applied force on the cable.
        cable (FibreLink): The cable object representing the fiber link.

    Returns:
        numpy.matrix: The polarisation rotation matrix.
    """

    # the difference between the fast axis and the slow axis
    # will be 90 degrees because we assume the wind hits the fibre only laterally
    rfi = 6 * 10**-5  # refractive index difference
    d = cable.cable_radius * 2  # diameter of the cable
    lbda = 400 * 10**-9  # wavelength of light = assume 400 nm
    delta = rfi * force / (lbda * d)  # birefringence
    gamma = delta * cable.cable_length  # phase retardation

    # Gamma = phase retardation
    rot = np.matrix([[np.exp(-1.0j * gamma / 2), 0], [0, np.exp(1.0j * gamma / 2)]])
    return rot
