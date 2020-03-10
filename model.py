# This file defines constants and parameters
# constants are fetched from https://nssdc.gsfc.nasa.gov/planetary/planetfact.html

import numpy as np


#
#  elements:
#    numpy.array:
#      0      - Semi-Major Axis (a)
#      1      - Eccentricity (e)
#      2      - Inclination (i)
#      3      - Right Ascension Point (ap)
#      4      - Perigee (omega)
#      5      - Mean Anomaly (M)
#



#
#   constant parameters of the prediction
#

# Coefficient of the Gravity
G = 6.67384E-11
AU = 1.496E+11

### Satellite Parameters:

# mass of the satellite
mass = 685.2  # kg
# drag coefficient: used in calculation of atmosphere drag force
cd = 2.2
# net sectional area: used in calculations of atmosphere drag and solar radiation pressure. Considered 1/4 of the total surface area.
sd = 3.63 # m^2
# reflective coefficient of the satellite, used in solar radiation pressure, usually between 1.0 ~ 1.44
cr = 1.44 


### Earth Parameters:

# mass of the earth
Mass = 5.9723E+24 # kg
GM = 0.3986004415E+15
# mean radius of the earth
R = 0.6378136300E+07 # m
# J2 coefficient of the earth
J2 = 1.0826361E-3 
# rotate angular speed of the earth
rotate = 7.2921159E-5 # rad/s

# atmosphere parameter:
A = 1.3 # kg/m3
H = 8E+3 # m


### Sun Parameters:

# mass of the Sun
ms = 1.9885E+30 # kg
Gms = 1.32712E+20
# coefficient of Sun shine
sr = 4.560e-6  


### Moon Parameters:

# mass of the Moon
mm = 7.346E+22 # kg
Gmm = 4.9E+12

#
# temperary variables
#   updated in predict.change()
class var:
    # position vector of the satellite
    r = np.zeros(3) # m
    # velocity vector of teh satellite
    v = np.zeros(3) # m/s
    # Semi-latus Rectum
    p = 0.0 # m
    # Laplace Vector
    e = np.zeros(3)
    # h Vector
    h = np.zeros(3)
    # True Anomaly
    f = 0.0 # rad
    # Change rate of Mean Anomaly
    nc = 0.0 # rad/s
    # Eccentrical Anomaly
    Ec = 0.0 # rad

    def __init__(self, r, v, p, e, h, f, nc, Ec):
        self.r = r
        self.v = v
        self.p = p
        self.e = e
        self.h = h
        self.f = f
        self.nc = nc
        self.Ec = Ec
