from typing import List
import numpy as np

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

def sinusoidal_control(t: float, params: List[np.array]):
    """
    This function creates a sinusoidal control signal.

    Parameters:
    t (float): The time at which the signal is evaluated.
    params (np.array): A numpy array containing the parameters of the sinusoidal signal.

    Returns:
    float: The value of the signal at time t.
    """
    # print(f"received params: {params}")
    _ret = []
    for _, param in enumerate(params):
        # print(f"processing {params[i]}")
        _ret.append(sinusoid(t, *param))
    # print(f"returning {_ret}")
    return _ret