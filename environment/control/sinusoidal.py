
"""
This module provides functions for generating and applying sinusoidal control signals.

The module contains the following functions:
- sinusoid(t, A, f, phi, offset=0.0): Creates a sinusoidal signal at a given time.
- sinusoidal_control(t, params): Samples the sinusoidal control sum at the
given time, for the provided parameters of the sinusoid params lists.
"""
from typing import List
import numpy as np

# pylint: disable=invalid-name
def sinusoid(
    t: float,
    A: float,
    f: float,
    phi: float,
    offset: float = 0.0,
):
    """
    This function creates a sinusoidal signal.

    Parameters:
    t (float): The time at which the signal is evaluated.
    A (float): The amplitude of the signal.
    f (float): The frequency of the signal.
    phi (float): The phase of the signal.
    offset (float): The offset of the signal.

    Returns:
    float: The value of the signal at time t.
    """
    return A * np.sin(2 * np.pi * f * t + phi) + offset
# pylint: enable=invalid-name

def sinusoidal_control(t: float, params: List[np.array]): # type: ignore
    """
    Applies sinusoidal control to the given time `t` using the provided parameters.

    Args:
        t (float): The time at which to apply the control.
        params (List[np.array]): A list of parameters for the sinusoidal control. Parameters:
            t (float): The time at which the signal is evaluated.
            A (float): The amplitude of the signal.
            f (float): The frequency of the signal.
            phi (float): The phase of the signal.
            offset (float): The offset of the signal.

    Returns:
        List[float]: A list of control values sampled from each sinsoid at time t.
    """
    _ret = []
    for _, param in enumerate(params):
        _ret.append(sinusoid(t, *param))
    return _ret
