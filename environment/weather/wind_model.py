import numpy as np

from ..core.fibre_cable import FibreLink
from ..core.polarisation_disturbance import polarisation_from_force

# density of air
AIR_DENSITY = 1.229  # kg/m^3 -- at sea level


class WindModel:
    def __init__(self, data):
        self.index = 0
        self.wind_speed = data["wind_speed"]
        self.wind_direction = data["wind_direction"]
    
    def reset(self):
        self.index = 0

    def compute_sample(self, fibre_cable: FibreLink):
        F = (
            1
            / 2
            * AIR_DENSITY
            * area_hit(self.wind_direction[self.index], fibre_cable)
            * self.wind_speed[self.index] ** 2
            * fibre_cable.drag_coefficient
        )  # N
        
        self.next_sample()
        return polarisation_from_force(F, fibre_cable)

    def next_sample(self):
        self.index += 1


def area_hit(wind_angle: float, fibre_cable: FibreLink):
    return (
        2
        * np.pi
        * fibre_cable.cable_radius
        * fibre_cable.cable_length
        * np.cos(wind_angle * np.pi / 180)
    )  # m^2 #! assume 0 degrees means wind is perpendicular to the cable


# t = np.linspace(0, 10)  # hours
# wd = data["wind_direction"]  # degrees
# ws = data["wind_speed"]  # km/h

# cable_length = 100  # m
# cable_radius = 0.01  # m

# cD = 0.5  # drag coefficient -- for a cylinder #! will differ for different cables
