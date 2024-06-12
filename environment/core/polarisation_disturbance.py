import numpy as np
from fibre_cable import FibreLink

def polarisation_from_force(force : float, cable : FibreLink):
    delta_n = 0.5
    # the difference between the fast axis and the slow axis
    # will be 90 degrees because we assume the wind hinds the fibre only laterally
    rfi = 6 * 10 ** -5 # refractive index difference
    d = cable.cable_radius * 2 # diameter of the cable
    lbda = 400 * 10 ** -9 # wavelength of light = assume 400 nm
    delta = rfi * force / (lbda * d) # birefringence
    gamma = delta * cable.cable_length # phase retardation

    # Gamma = phase retardation
    rot = np.array([[np.exp(-1.0j * gamma / 2), 0], [0, np.exp(1.0j * gamma / 2)]])
    return rot