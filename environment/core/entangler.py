"""
This module contains a function for entangling a given input state.

The entangler function takes a state matrix as input and rearranges
its elements to create an entangled state.
"""

import numpy as np


def entangler(input_state: np.matrix):
    """
    Entangles the input state by rearranging its elements.

    Parameters:
    inputState (numpy.matrix): The input state matrix with shape (2, 1).

    Returns:
    numpy.matrix: The entangled state matrix with shape (4, 1).
    """
    return np.matrix([[input_state[0, 0]], [0], [0], [input_state[1, 0]]])
