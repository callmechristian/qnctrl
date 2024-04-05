"""
This module contains functions for calculating the probability of detecting a specific
outcome in a quantum entanglement experiment, the Quantum Bit Error Rates (QBERs) for
n entangled state, and the probability of detecting a specific outcome in a quantum
entanglement experiment with noise.
"""

import numpy as np


def settings(basis: int, state: int):
    """
    Returns the measurement settings based on the given basis and state.

    Parameters:
    Basis (int): The basis value (0 or 1).
    State (int): The state value (0 or 1).

    Returns:
    numpy.matrix: The measurement settings based on the given basis and state.
    """
    if basis == 0:
        if state == 0:
            return np.matrix([[1, 0], [0, 0]])
        return np.matrix([[0, 0], [0, 1]])
    if state == 0:
        return 1 / 2 * np.matrix([[1, 1], [1, 1]])
    return 1 / 2 * np.matrix([[1, -1], [-1, 1]])


def prob_det(
    entangled_state: np.matrix, basis_a: int, state_a: int, basis_b: int, state_b: int
):
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
    # Calculate the measurement settings for Alice and Bob
    alice = settings(basis_a, state_a)
    bob = settings(basis_b, state_b)

    # Calculate the density matrix rho of the entagled state
    rho = entangled_state @ entangled_state.H

    # Calculate the probability of detecting the specified outcome
    return np.abs(np.trace(np.kron(alice, bob) @ rho))


def compute_qber(entangled_state: np.matrix):
    """
    Calculate the Quantum Bit Error Rates (QBERs) for an entangled state.

    Parameters:
    entangledState (np.matrix [4,1]): The entangled state for which to calculate the QBERs.

    Returns:
    list: A list containing the QBERz and QBERx values i.e. [QBERz, QBERx]

    """
    qber_z = prob_det(entangled_state, 0, 0, 0, 1) + prob_det(
        entangled_state, 0, 1, 0, 0
    )
    qber_x = prob_det(entangled_state, 1, 0, 1, 1) + prob_det(
        entangled_state, 1, 1, 1, 0
    )
    return [qber_z, qber_x]


def compute_probability_density(rho, basis_a, state_a, basis_b, state_b):
    """
    Compute the probability density of a measurement outcome given the input
    states and measurement bases.

    Parameters:
    rho (numpy.ndarray): The density matrix representing the quantum state.
    basis_a (numpy.ndarray): The measurement basis for Alice's measurement.
    state_a (numpy.ndarray): The quantum state for Alice's measurement.
    basis_b (numpy.ndarray): The measurement basis for Bob's measurement.
    state_b (numpy.ndarray): The quantum state for Bob's measurement.

    Returns:
    float: The probability density of the measurement outcome.
    """
    alice = settings(basis_a, state_a)
    bob = settings(basis_b, state_b)
    return np.abs(np.trace(np.kron(alice, bob) @ rho))


def compute_all_probability_densities(rho, ratio_zx):
    """
    Compute the probability densities for all possible measurement outcomes.

    Args:
        rho (numpy.ndarray): The density matrix representing the quantum state.
        ratio_zx (float): The ratio of the Z-basis measurement probability to the X-basis
        measurement probability.

    Returns:
        numpy.ndarray: A 4x4 array containing the probability densities for all possible
        measurement outcomes.
    """
    probability = [ratio_zx, 1 - ratio_zx]
    all_probabilities = np.zeros([4, 4])
    for i in range(16):
        basis_a = i >> 1 & 1
        state_a = i & 1
        basis_b = i >> 3 & 1
        state_b = i >> 2 & 1

        all_probabilities[basis_a * 2 + state_a, basis_b * 2 + state_b] = (
            compute_probability_density(rho, basis_a, state_a, basis_b, state_b)
            * probability[basis_a]
            * probability[basis_b]
        )
    return np.array(all_probabilities)


def compute_noisy_qber(
    entangled_state: np.matrix,
    noise: float = 0.02,
    ratio_zx: float = 0.7,
    n_z: int = 2**14,
):
    """
    Computes the noisy statistics of a given entangled state.

    Args:
        entangled_state (np.matrix): The input entangled state.
        noise (float, optional): The noise level. Defaults to 0.02.
        ratio_zx (float, optional): The ratio between Z and X measurements. Defaults to 0.7.
        n_z (int, optional): The number of measurements. Defaults to 2**14.

    Returns:
        list: A list containing the quantum bit error rates for Z and X measurements, respectively.
    """
    # Compute the density matrix of the input entangled state
    rho_entangled_state = entangled_state @ entangled_state.H

    # Create the identity matrix
    rho_id = np.matrix(np.identity(4)) / 4.0

    # Combine the entangled state and identity matrix with noise
    rho = (1 - noise) * rho_entangled_state + noise * rho_id

    # Compute the probability densities for all possible measurement outcomes
    all_probabilities = compute_all_probability_densities(rho, ratio_zx)

    # Extract the probability densities for Z measurements
    z_probabilities = all_probabilities[:2, :2]

    # Extract the probability densities for X measurements
    # x_probabilities = all_probabilities[2:4, 2:4] # ? not used

    # Compute the number of runs based on the total number of measurements
    # and the sum of probabilities for Z measurements
    number_of_runs = n_z / z_probabilities.sum()

    # Generate a random number of detections based on the probabilities for all measurements
    random_detections = np.random.poisson(number_of_runs * all_probabilities)

    # Adjust the number of detections for the last element to match the total number of measurements
    random_detections[1, 1] += n_z - random_detections[:2, :2].sum()

    # Extract the number of detections for Z measurements
    z_detections = random_detections[:2, :2]

    # Extract the number of detections for X measurements
    x_detections = random_detections[2:4, 2:4]

    # Compute the quantum bit error rate for Z measurements
    qber_z = (z_detections[0, 1] + z_detections[1, 0]) / z_detections.sum()

    # Compute the quantum bit error rate for X measurements
    qber_x = (x_detections[0, 1] + x_detections[1, 0]) / x_detections.sum()

    # Return the quantum bit error rates for Z and X measurements as a list
    return [qber_z, qber_x]
