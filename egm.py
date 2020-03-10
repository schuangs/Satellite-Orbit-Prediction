# This file calculate the Eathr Gravity Model Effect using Spherical Harmonic Polynomials Method

import model
from highPrecision import GCRS2ITRS, ITRS2GCRS

import numpy as np
from scipy import special
import re
from astropy import coordinates as coord


# Gravity acceleration in Spherical Coordinate
# see http://icgem.gfz-potsdam.de/str-0902-revised.pdf, Section 4.4


# read .gfc file, 
# return the gravity accelration function
def readGFC(filename):
    # read .gfc file
    foo = open(filename, mode="r")
    start = False
    for line in foo:
        if line[0:10] == "max_degree":
            s = re.split(r'\s+', line)
            n = int(s[1]) # n is the degree of expansion
            C = np.zeros((n+1, n+1))
            S = np.zeros((n+1, n+1))
        if line[0:11] == "end_of_head":
            start = True
            continue
        if start == True:
            s = re.split(r'\s+', line)
            l,m = int(s[1]), int(s[2])
            C[l][m] = float(s[3])
            S[l][m] = float(s[4])
    foo.close()
    # create the function of gravity acceleration
    #   init_jd : initial Julian Date
    #   t : current time from the start in second
    def egmF(elements, var, init_jd, t, order_used, lc):
        # translate positions and velocities from GCRS into ITRS frame
        r1, v1 = GCRS2ITRS(var.r, var.v, init_jd, t/24./3600.)
        # translate cartesian coordinates into spherical coordinates of positions in ITRS frame
        r, phi, lamda = coord.cartesian_to_spherical(r1[0], r1[1], r1[2])
        phi, lamda = phi.value, lamda.value
        # get the values of Legendre Polynomials and their derivatives
        Pmn, Pmn_d = special.lpmn(n, n, np.sin(phi))
        Fsc = np.zeros(3)
        for l in range(1, order_used+1):
            for m in range(l+1):
                Fsc[0] += -model.GM/r/r * (model.R/r)**l * (l+1) * Pmn[m,l]*lc[m, l] * (C[l,m]*np.cos(m*lamda) + S[l, m]*np.sin(m*lamda))
                Fsc[1] += model.GM/r * (model.R/r)**l * m * Pmn[m,l]*lc[m, l] * (S[l, m]*np.cos(m*lamda) - C[l,m]*np.sin(m*lamda))
                Fsc[2] += model.GM/r * (model.R/r)**l * Pmn_d[m,l]*lc[m, l] * np.cos(phi) * (C[l,m]*np.cos(m*lamda) + S[l, m]*np.sin(m*lamda))
        Fsc[1] /= r*np.cos(phi)
        Fsc[2] /= r
        # translate the force from spherical to cartesian in ITRS frame
        Fxyz = force_spherical_to_cartesian(Fsc, lamda, phi)
        # translate the force from ITRS to GCRS frame
        FF, v = ITRS2GCRS(Fxyz, Fxyz, init_jd, t/24./3600.)
        # matrix transform from Spherical Coordinates to Cartesian Coordinates
        return FF
    return egmF


# normalization coefficients of Fully Associated Legendre Polynomials with degree l, order m
def normalize(m, l):
    norm = (2*l+1)
    if m != 0:
        norm *= 2
    for i in range(l-m+1, l+m+1):
        norm /= i
    return np.sqrt(norm)

# get a matrix containing coefficients of Fully Associated Legendre Polynomials with degree 0...n and order 0...n
def LegendreCoef(n):
    lc = np.zeros((n+1, n+1))
    for l in range(n+1):
        for m in range(l+1):
            lc[m, l] = normalize(m, l)
    return lc

# translate force from spherical to cartesian coordinates
#   Fsc      - force vector in spherical coordinate, [Fr, Flamda, Fphi]
#   lamda    - longitude, [0, 2*PI)
#   phi      - lattitude (from equator), [-PI/2, PI/2]
def force_spherical_to_cartesian(Fsc, lamda, phi):
    matrix = np.array([
        [np.cos(phi)*np.cos(lamda), -np.sin(lamda), -np.sin(phi)*np.cos(lamda)],
        [np.cos(phi)*np.sin(lamda), np.cos(lamda), -np.sin(phi)*np.sin(lamda)],
        [np.sin(phi), 0, np.cos(phi)]
    ])
    return np.matmul(matrix, Fsc)