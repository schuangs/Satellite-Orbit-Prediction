# This file contains some high precision algorithms for orbit prediction

from sgp4.api import Satrec
from sgp4.api import jday
import skyfield.sgp4lib as sgp4lib
from astropy import coordinates as coord, units as u
from astropy.time import Time 

import numpy as np

#
# function sgp4:
#   SGP4 algorithm, see https://pypi.org/project/sgp4/
#   input:
#     s, t        - TLE string of the satellite, http://celestrak.com/NORAD/elements/active.txt
#     jd          - Julian date
#     fr          - Second fraction of the day
#   return:
#     r           - position coordinate in ITRS frame
#     v           - velocity coordinate in ITRS frame
#
def sgp4Predict(s, t, jd, fr):
  satellite = Satrec.twoline2rv(s, t)
  jd += 0.5
  # get the position and velocity vectors in TEME frame
  e, r, v = satellite.sgp4(jd, fr)
  # Conversion from TEME to ITRS
  r,v= sgp4lib.TEME_to_ITRF(jd+fr,np.asarray(r),np.asarray(v)*86400)
  v=v/86400
  return r*1000, v*1000


# translate from ITRS frame to GCRS frame, velocity and positoin should be considered differently
def ITRS2GCRS(r, v, jd, fr):
  now = Time(jd+fr, format="jd")
  itrs = coord.ITRS(x=r[0]*u.m, y=r[1]*u.m, z=r[2]*u.m, v_x=v[0]*u.m/u.s, v_y=v[1]*u.m/u.s, v_z=v[2]*u.m/u.s, obstime=now, representation_type='cartesian')
  gcrs = itrs.transform_to(coord.GCRS(obstime=now))
  r, v = gcrs.cartesian.xyz.value, gcrs.velocity.d_xyz.value
  return r, v*1000

# translate from GCRS frame to ITRS frame, velocity and position should be considered differently
def GCRS2ITRS(r, v, jd, fr):
  now = Time(jd+fr, format="jd")
  itrs = coord.GCRS(r[0]*u.m, r[1]*u.m, r[2]*u.m, v[0]*u.m/u.s, v[1]*u.m/u.s, v[2]*u.m/u.s, obstime=now, representation_type='cartesian')
  gcrs = itrs.transform_to(coord.ITRS(obstime=now))
  r, v = gcrs.cartesian.xyz.value, gcrs.velocity.d_xyz.value
  return r, v*1000