import sympy as sp
import math
import numpy as np


class WindModel:
    def __init__(self, data):
        self.index = 0
        self.wind_speed = data['wind_speed']
        self.wind_direction = data['wind_direction']
    
    def compute_next_sample(self):
        F = self.wind_speed[t], self.wind_direction[t]


t = np.linspace(0, 10) # hours
wd = data['wind_direction'] # degrees
ws = data['wind_speed'] # km/h

cable_length = 100 # m
cable_radius = 0.01 # m

cD = 0.5 # drag coefficient -- for a cylinder #! will differ for different cables

air_density =  1.229 # kg/m^3 -- at sea level

def area_hit(wind_angle):
    return 2 * np.pi * cable_radius * cable_length * np.cos(wind_angle * np.pi / 180) # m^2 #! assume 0 degrees means wind is perpendicular to the cable

mass_of_air = air_density * area_hit(wd) * ws**2 * cable_length # kg

F = 1/2 * air_density * ws**2 * area_hit(wd) * cD # N