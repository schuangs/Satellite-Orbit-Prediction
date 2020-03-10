# This file contains functions of some transformations and mathematical calculations

import model
import highPrecision

import numpy as np
import re
import scipy.interpolate

# Numerically solute the Kepler Equation, using Newton iteration
def KeplerEquation(M, e):
    # precision of the calculation
    eps = 0.00001
    E = M
    E1 = 0.
    while E-E1 > eps or E-E1 < -eps:
        E1 = E
        E = E1 - (E1-e*np.sin(E1)-M)/(1.-e*np.cos(E1))
    return E

# calculate the tranform matrix from RTN system to ECI system
#   theta is the angle of f + angle of perigee, which means the angle between r vector and n vector
def RTN2ECI(theta, ap, i):
    return np.array([[np.cos(theta)*np.cos(ap)-np.sin(theta)*np.sin(ap)*np.cos(i), -np.sin(theta)*np.cos(ap)-np.cos(theta)*np.sin(ap)*np.cos(i), np.sin(i)*np.sin(ap)],
                     [np.cos(theta)*np.sin(ap)+np.sin(theta)*np.cos(ap)*np.cos(i), -np.sin(theta)*np.sin(ap)+np.cos(theta)*np.cos(ap)*np.cos(i), -np.sin(i)*np.cos(ap)],
                     [np.sin(theta)*np.sin(i), np.cos(theta)*np.sin(i), np.cos(i)]])

# get var from elements
# transform from elements to states(positions and velocities) as well
def getVar(elements):
    mu = model.GM
    # change rate of Mean Anomaly
    nc = np.sqrt(mu/elements[0]**3)

    # Eccentrical Anomaly
    Ec = KeplerEquation(elements[5], elements[1])
    # True Anomaly
    if Ec == np.pi:
        f = np.pi
    else:
        f = 2*np.arctan(np.sqrt((1.+elements[1])/(1.-elements[1])) * np.tan(Ec/2))
        if Ec > np.pi:
            f += 4*np.pi
        
    # Semi-latus Rectum
    p = elements[0]*(1-elements[1]**2)
    # length of h vector
    H = np.sqrt(mu*p)
    # velocity components
    vt = H/p*(1.+elements[1]*np.cos(f))
    vr = H/p*elements[1]*np.sin(f)
    L = p / (1.+elements[1]*np.cos(f))

    # calculate transformation matrix from RTN to ECI system
    trans = RTN2ECI(f + elements[4], elements[3], elements[2])

    # transform position and velocity from RTN to ECI system
    r = np.matmul(trans, np.array([L, 0., 0.]))
    v = np.matmul(trans, np.array([vr, vt, 0.]))

    # h vector
    h = np.cross(r, v)
    # Laplace vector
    e = 1./mu * np.cross(v, h) - r / L

    return model.var(r, v, p, e, h, f, nc, Ec)

# routine to read CPF file, return positions coordinates array (in ECEF system) and time array
# and also read Julian Dates and Second Fractions
def readCPF(filename, num=100):
    foo = open(filename, mode='r')
    count = 0
    positions = []
    interval = 0.0
    jds = []    # Julian dates
    seconds = []    # second fractions
    for line in foo:
        if count == num:
            break
        # abandon lines start with 'H'
        if line[0] == 'H':
            if line[1] == '2':
                s = re.split(r'\s+', line)
                interval = float(s[-7])
            continue
        s = re.split(r'\s+', line)
        positions.append([float(s[-4]), float(s[-3]), float(s[-2])])
        jds.append(float(s[2]) + 2400000)
        seconds.append(float(s[3])/3600/24)
        count += 1
    times = np.linspace(0.0, (num-1)*interval, num)
    foo.close()
    return np.array(positions), times, np.array(jds), np.array(seconds)

# Lagrange Interpolation
def LagrangeInterpolation(xs, ys):
    n = xs.size
    # derivative of the interpolation function
    der1 = scipy.interpolate.lagrange(xs, ys[:,0]).deriv()
    der2 = scipy.interpolate.lagrange(xs, ys[:,1]).deriv()
    der3 = scipy.interpolate.lagrange(xs, ys[:,2]).deriv()
    def der(x):
        return np.array([der1(x), der2(x), der3(x)])
    return der

# get velocity vectors from position vectors through n-order Lagrange Interpolation
# and trim the data: throw the data near the start and end
# n is the number of points used to interpolate each point
def getData(positions, times, jds, seconds, n=9):
    m = times.size
    rs = []
    for i in range(m):
        r, v = highPrecision.ITRS2GCRS(r = positions[i], v = np.zeros(3), jd = jds[i], fr = seconds[i])
        rs.append(r)
    rs = np.array(rs)
    velocity = []
    for i in range(n//2, m-n//2):
        der = LagrangeInterpolation(times[i-n//2: i+n//2+1], rs[i-n//2: i+n//2+1])
        velocity.append(der(times[i]))
    return rs[n//2: m-n//2], np.array(velocity), times[n//2: m-n//2], jds[n//2]

# transform from states(position and velocity vectors) to elements
def states2Elements(r, v):
    mu = model.GM
    # h vector
    h = np.cross(r, v)
    # length of h vector
    H = np.linalg.norm(h)
    # length of r vector
    L = np.linalg.norm(r)
    # length of v vector
    u = np.linalg.norm(v)
    # Semi-Major Axis
    a = 1/(2./L - u**2/mu)
    # Semilatus Rectum
    p = H**2 / mu
    # Laplace constant vector
    e = 1./mu * np.cross(v, h) - r / L
    # Eccentricity
    E = np.sqrt(1-p/a)
    # Orbit Inclination
    i = np.arccos(h[2] / H)

    # ap is the Ascension Point Angle, whose quadrant should be decided according to h vector
    if h[0] == 0:
        if h[1] > 0:
            ap = np.pi
        elif h[1] < 0:
            ap = 0
    else:
        ap = np.arctan(h[1]/h[0])
        if h[0] < 0:
            ap += np.pi
        ap += np.pi/2.
    
    # n is the normal vector from origin to ascension point
    temp = -np.cross(h, np.array([0., 0., 1.]))
    n = temp / np.linalg.norm(temp)

    # omega is the angle of perigee, which should be determined by h vector and n vector
    if h[2] * np.cross(n, e)[2] > 0:
        omega = np.arccos((n[0]*e[0] + n[1]*e[1] + n[2]*e[2])/E)
    else:
        omega = -np.arccos((n[0]*e[0] + n[1]*e[1] + n[2]*e[2])/E) + 2*np.pi

    # f is the True Anomaly, which should be determined by Laplace vector and r vector
    if h[2] * np.cross(e, r)[2] > 0:
        f = np.arccos((e[0]*r[0] + e[1]*r[1] + e[2]*r[2])/L/E)
    else:
        f = -np.arccos((e[0]*r[0] + e[1]*r[1] + e[2]*r[2])/L/E) + 2*np.pi
    # Ec is the Eccentrical Anomaly, which should be determined by f
    if f == np.pi:
        Ec = np.pi
    else:
        Ec = 2*np.arctan(np.sqrt((1.-E)/(1.+E)) * np.tan(f/2))
        if f > np.pi:
            Ec += 4*np.pi
    
    # M is the Mean Anomaly, calculated directly through Kepler Equation
    M = Ec - E * np.sin(Ec)
    # all of the elements are calculated, and some more parameters are passed for convenience
    return np.array([a, E, i, ap, omega, M])

# transform the force in XYZ coordinates(Fx, Fy, Fz) into RTN coordinates(S, T, W)
def ForceGCRS2RTN(F, var):
    L = np.linalg.norm(var.r)
    S = np.dot(var.r, F)/ L
    T = np.dot(np.cross(np.cross(var.r, var.v), var.r)/(np.linalg.norm(np.cross(var.r, var.v))*L), F)
    W = np.dot(np.cross(var.r, var.v), F) / np.linalg.norm(np.cross(var.r, var.v))
    return np.array([S, T, W])