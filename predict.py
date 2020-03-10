# Functions for orbit prediction

import model
import perturbation
import transform

import numpy as np
from scipy.integrate import solve_ivp


#
# function predict:
#   input:
#     elements          - elements of satellite orbit, defined in model.py
#     ts                - array of time points
#     flag              - whether to consider perturbation
#   return:
#     list of elements at each predict point
#
def predict(elements, ts, egmF, init_jd, order_used, lc):
    return solve_ivp(change, t_span=(ts[0], ts[-1]), y0=elements, method="RK45", t_eval=ts, args=(egmF, init_jd, order_used, lc, ), max_step=100)


#
# functoin change:
#   input:
#     elements        - elements of orbit
#     t               - current time
#     flag            - whether consider perturbations
#   return:
#     derivatives of elements
#
def change(t, elements, egmF, init_jd, order_used, lc):
    print("t = ", t, " s")
    var = transform.getVar(elements)

    # perturbations
    derivative = perturbation.effect(elements, var, egmF, init_jd, t, order_used, lc)
    # derivative = np.zeros(6)
    derivative[5] += var.nc
    # derivative[4] = 0
    # print(derivative)
    return derivative