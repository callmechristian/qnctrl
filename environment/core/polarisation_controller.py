"""
This module provides a function for creating a control gate
for a quantum system using a series of phase shifts and rotations.
"""

import numpy as np
from numpy import cos, sin, pi

# type: ignore is used to ignore the type hinting for mypy (doesn't support null default values)
def polar_control(
    phis: np.array = None, # type: ignore
    phi1: float = None, # type: ignore
    phi2: float = None, # type: ignore
    phi3: float = None, # type: ignore
    phi4: float = None, # type: ignore
):
    """
    This function creates a control gate for a quantum system using
    a series of phase shifts and rotations.

    Parameters:
    phi1, phi2, phi3, phi4 (float): The phase angles in radians.

    Returns:
    np.matrix: A 2x2 matrix that represents the control gate.
    """
    # if not arguments supplied, return error
    if phis is None and (phi1 is None or phi2 is None or phi3 is None or phi4 is None):
        raise ValueError("At least all phase angles must be supplied.")

    # Rotation matrix for a positive 45 degree rotation
    rotp45 = np.matrix(
        [[cos(pi / 4.0), sin(pi / 4.0)], [-sin(pi / 4.0), cos(pi / 4.0)]]
    )

    # Rotation matrix for a negative 45 degree rotation
    rotm45 = np.matrix(
        [[cos(-pi / 4.0), sin(-pi / 4.0)], [-sin(-pi / 4.0), cos(-pi / 4.0)]]
    )

    def phase(phi: float):
        """
        This function creates a phase shift matrix for a given phase angle.

        Parameters:
        phi (float): The phase angle in radians.

        Returns:
        np.matrix: A 2x2 phase shift matrix.
        """
        return np.matrix([[1, 0], [0, np.exp(1.0j * phi)]])

    if phis is not None:
        # print(f"phis: {phis} phis[0]: {phis[0]}")
        if len(phis) != 4:
            phis = phis[0]
        [phi1, phi2, phi3, phi4] = phis
    return (
        phase(phi1)
        @ rotp45
        @ phase(phi2)
        @ rotm45
        @ phase(phi3)
        @ rotp45
        @ phase(phi4)
        @ rotm45
    )
