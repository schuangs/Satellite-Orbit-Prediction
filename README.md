# Satellite-Orbit-Prediction
Programs for the prediction of satellite orbit in Python.

@author Junkang Huang   huangjk8@mail2.sysu.edu.cn





*simulation.py*:

​	Main file for the prediction.

​	Run the following order in a console:

> python simulation.py

​	First enter the relative path of the CPF file of a satellite, then enter the number of data points you want to use (should be smaller than total number of  data in the file).

​	Or you can choose to use SGP4 algorithm with TLE input to get a higher precision prediction.

​	**Requirement**:

* `Python 3` environment
* `numpy`, `scipy`, `matplotlib` packages need to be installed
* `sgp4`,  `skyfield`, `astropy` packages installed



*model.py* :

​	Defines parameters, constants and important data structure

* `elements`: `(6,) numpy.array`:

  \#   0   - *Semi-Major Axis (a)*

  \#   1   - *Eccentricity (e)*

  \#   2   - *Inclination (i)*

  \#   3   - *Right Ascension Point (ap)*

  \#   4   - *Perigee (omega)*

  \#   5   - *Mean Anomaly (M)*

* basic constants

* parameters of the satellite

* parameters of the Earth

* parameters of the Sun

* parameters of the Moon

* class `var`: some temporary variables



*transform.py* : 

​	Defines transformation or mathematical calculation functions

* `KeplerEquation() `: Numerical solution of the Kepler Equation
* `RTN2ECI()` : get the transform matrix from **RTN** system to **ECI** system

* `ECEF2ECI()`: get the transform matrix from **ECEF** system to **ECI** system
* `getVar()`: transform from `elements` to satellite states(position and velocity vectors)
* `readCPF()`: routine to read a CPF file (.jax)
* `LagrangeInterpolation()`: Lagrange interpolation algorithm implementation
* `getData()`: CPF files only contain position information. Thus we need to get the velocities using interpolation. And some input data are abandoned near the start and end of the input.
* `states2Elements()`: transform from states(position and velocity vectors) to `elements`



*predict.py*

​	Functions used for prediction, which are called in *simulation.py*

* `predict()`: input initial `elements` and time points, then this function integrates using `scipy.solve_ivp()` with *DOP853* integration algorithm
* `change()`: update function of `elements`, used in `scipy.solve_ivp()`.



*perturbation.py*

​	Perturbations considered in the prediction. This determines the precision of our prediction.

* `effect()`: total perturbation function
* `nonspherical()`: the *non-spherical perturbation* of the Earth, only *J2* item considered.
* `atmospherical()`: the *atmospherical drag perturbation* of the atmosphere.
* `pressure()`: the *solar radiation pressure perturbation* of the Sun
* `thirdBody()`: calculate the *third body acceleration effect* of a third body(the Sun and the Moon)
* `totalThirdBody()`: consider the *third body effect* of the Sun and the Moon
* `forceAnalysis()`: calculate the effect on `elements` from the force on the satellite.



*highPrecision.py*

​	Some existing algorithms for orbit prediction.

* `sgp4Predict()`: using SGP4 algorithm
