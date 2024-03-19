import numpy as np

def ProbDet(entangledState: np.matrix, BasisA: int, StateA: int, BasisB: int, StateB: int):
    """
    Calculates the probability of detecting a specific outcome in a quantum entanglement experiment.

    Parameters:
    entangledState (numpy.matrix [4,1]): The entangled state of the system.
    BasisA (int): The basis used by Alice for measurement.
    StateA (int): The state used by Alice for measurement.
    BasisB (int): The basis used by Bob for measurement.
    StateB (int): The state used by Bob for measurement.

    Returns:
    float: The probability of detecting the specified outcome.

    """
    def settings(Basis: int, State: int):
        """
        Returns the measurement settings based on the given basis and state.

        Parameters:
        Basis (int): The basis value (0 or 1).
        State (int): The state value (0 or 1).

        Returns:
        numpy.matrix: The measurement settings based on the given basis and state.
        """
        if (Basis==0):
            if(State==0):
                return np.matrix([[1,0],[0,0]])
            else:
                return np.matrix([[0,0],[0,1]])
        else:
            if(State==0):
                return 1/2*np.matrix([[1,1],[1,1]])
            else:
                return 1/2*np.matrix([[1,-1],[-1,1]])
    # Calculate the measurement settings for Alice and Bob
    Alice = settings(BasisA, StateA)
    Bob = settings(BasisB, StateB)
    
    # Calculate the density matrix rho of the entagled state
    rho = entangledState @ entangledState.H
    
    # Calculate the probability of detecting the specified outcome
    return np.abs(np.trace(np.kron(Alice, Bob) @ rho))

def QBERs(entangledState: np.matrix):
    """
    Calculate the Quantum Bit Error Rates (QBERs) for an entangled state.

    Parameters:
    entangledState (np.matrix [4,1]): The entangled state for which to calculate the QBERs.

    Returns:
    list: A list containing the QBERz and QBERx values i.e. [QBERz, QBERx]

    """
    QBERz = ProbDet(entangledState, 0, 0, 0, 1) + ProbDet(entangledState, 0, 1, 0, 0)
    QBERx = ProbDet(entangledState, 1, 0, 1, 1) + ProbDet(entangledState, 1, 1, 1, 0)
    return [QBERz, QBERx]