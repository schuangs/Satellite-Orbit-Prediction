# This file contains perturbation effects on the satellite motion

from transform import RTN2ECI, ForceGCRS2RTN
import model
import egm

import numpy as np
from astropy.coordinates import get_moon, get_sun
from astropy.time import Time 

# tota effect of all the perturbation
#   return the derivative increasement of elements
def effect(elements, var, egmF, init_jd, t, order_used, lc):
    return (
            nonspherical(elements, var, egmF, init_jd, t, order_used, lc)
            +atmospherical(elements, var)
            +pressure(elements, var, init_jd, t)
            +totalThirdBody(elements, var, init_jd, t)
           )

# nonspherical perturbation, only J2 item considered
def nonspherical(elements, var, egmF, init_jd, t, order_used, lc):
    # get the force in cartesian coordinates in GCRS frame
    F = egmF(elements, var, init_jd, t, order_used, lc)
    # translate the force from XYZ coordinates into RTN coordinates
    Fnew = ForceGCRS2RTN(F, var)
    return GaussianI(Fnew[0], Fnew[1], Fnew[2], elements, var)

# calculate the density of atmosphere
# based on the model of 
def getAtmosphereDensity(elements, var):
    return model.A * np.exp(-(elements[0]-model.R)/model.H) * np.exp(elements[0]*elements[1]*np.cos(var.Ec)/model.H)

# atmospherical perturbation, earth atmosphere is regarded constant
def atmospherical(elements, var):
    H = np.linalg.norm(var.h)
    L = np.linalg.norm(var.r)
    # velocity components
    vt = H/var.p*(1.+elements[1]*np.cos(var.f))
    vr = H/var.p*np.sin(var.f)
    v = np.sqrt(vt**2+vr**2)

    # coefficients of atmospherical drag force
    coef = 1./2.*model.cd*model.sd/model.mass*getAtmosphereDensity(elements, var)*v

    # components of drag force in three directions
    fr = -coef*vr
    ft = -coef*(vt-model.rotate*L*np.cos(elements[2]))
    fh = -coef*model.rotate*L*np.cos(var.f + elements[4])*np.sin(elements[2])

    # finally we need to transform the drag force to the effect on elements
    return GaussianI(fr, ft, fh, elements, var)

# calculate whether the satellite is exposed to the sun
def exposed(rsu, r):
    if np.dot(rsu, r) > 0:
        return 1.
    if np.linalg.norm(np.cross(rsu, r)) > model.R:
        return 1.
    return 0.

# solar radiation pressure
def pressure(elements, var, init_jd, t):
    L = np.linalg.norm(var.r)
    # position of the Sun
    rs = get_sun(Time(init_jd+t/24/3600, format="jd")).cartesian.xyz.value*model.AU
    # unit vector of rs
    rsu = rs / np.linalg.norm(rs)
    # unit vector of r
    ru = var.r / L
    # unit vector of h
    hu = var.h / np.linalg.norm(var.h)
    # exposure factor: whether the satellite is exposed to the sun, used in solar radiation pressure
    k = exposed(rsu, var.r)
    # length of the pressure force
    F = -k * model.sr * model.cr * model.sd / model.mass

    # three cosines
    c1 = np.dot(rsu, ru)
    c3 = np.dot(rsu, hu)
    c2 = 1. - c1**2 - c3**2

    # components of the force
    fr = F * c1
    ft = F * c2
    fh = F * c3

    return GaussianI(fr, ft, fh, elements, var)

# Gravitation of third body
def thirdBody(thirdBodyMass, rs, elements, var):
    d = var.r - rs
    F = -model.G * thirdBodyMass * (d/np.linalg.norm(d)**3 + rs/np.linalg.norm(rs)**3)
    # transform matrix from ECI system to RTN
    trans = np.linalg.inv(RTN2ECI(var.f + elements[4], elements[3], elements[2]))
    # third body gravitation force vector in orbit system
    return np.matmul(trans, F) 

# Third body perturbation of the Sun and the Moon
def totalThirdBody(elements, var, init_jd, t):
    rs = get_sun(Time(init_jd+t/24/3600, format="jd")).cartesian.xyz.value*model.AU
    rm = get_moon(Time(init_jd+t/24/3600, format="jd")).cartesian.xyz.value*1000
    # third body force vector in RTN system
    ff = np.array([0., 0., 0.])
    # Solar
    ff += thirdBody(model.ms, rs, elements, var)
    # Lunar
    ff += thirdBody(model.mm, rm, elements, var)
    return GaussianI(ff[0], ff[1], ff[2], elements, var)


# Gaussian First Perturbation Equation
def GaussianI(fr, ft, fh, elements, var):
    L = np.linalg.norm(var.r)
    # dap is the change rate of ap. Pick it out just for convenience.
    dap = L*np.sin(var.f + elements[4])*fh/(var.nc*elements[0]**2*np.sqrt(1-elements[1]**2)*np.sin(elements[2]))
    # directly use transform equations    
    return np.array([
        2./var.nc/np.sqrt(1.-elements[1]**2)*(elements[1]*np.sin(var.f)*fr + (1.+elements[1]*np.cos(var.f))*ft),
        np.sqrt(1.-elements[1]**2)/var.nc/elements[0] * (np.sin(var.f)*fr + (np.cos(var.f)+np.cos(var.Ec))*ft),
        L*np.cos(var.f + elements[4])*fh/(var.nc*elements[0]**2*np.sqrt(1.-elements[1]**2)),
        dap,
        np.sqrt(1.-elements[1]**2)/(var.nc*elements[0]*elements[1])*(-np.cos(var.f)*fr + (1.+L/var.p)*np.sin(var.f)*ft)-np.cos(elements[2])*dap,
        - (1.-elements[1]**2)/var.nc/elements[0]/elements[1]*((2*elements[1]*L/var.p-np.cos(var.f))*fr + (1.+L/var.p)*np.sin(var.f)*ft)
    ])