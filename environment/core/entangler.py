import numpy as np

def entangler(inputState: np.matrix):
    """
    Entangles the input state by rearranging its elements.

    Parameters:
    inputState (numpy.matrix): The input state matrix with shape (2, 1).

    Returns:
    numpy.matrix: The entangled state matrix with shape (4, 1).
    """
    return np.matrix([[inputState[0,0]],
               [0],
               [0],
               [inputState[1,0]]])