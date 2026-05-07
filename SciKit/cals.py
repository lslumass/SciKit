"""
minor functions for quick calculations and conversions
"""


import numpy as np

# constants
NA = 6.02214076e23  # Avogadro's number
kB = 1.380649e-23  # Boltzmann constant in J/K
R = 8.314462618  # Gas constant in J/(mol*K)


def cal_csat(number, box_length, droplet_radius=0):
    """
    calculate the concentration of dilute phase in a box with a droplet
    V_dilute = box_length^3 - 4/3 * pi * droplet_radius^3
    csat = number / V_dilute/NA*1e30    in uM
    """

    V_dilute = box_length**3 - 4 / 3 * np.pi * droplet_radius**3
    csat = number / V_dilute / NA * 1e30
    
    return csat
