import predict
import model
import transform
import highPrecision
import egm

import numpy as np
import matplotlib.pyplot as plt


# main function of prediction
def main():
    addr = input("Enter the relative address of satellite CPF file:\n")
    n = int(input("Enter the number of record points used:\n"))
    # read file
    positions, times, jds, seconds = transform.readCPF(addr, num = n+8)

    # decide whether using SGP4 algorithm
    usingSGP4 = False
    if usingSGP4:
        n = jds.size
        line1 = input("TLE line 1:\n")
        line2 = input("TLE line 2:\n")
        d_err = []
        r_p = []

        for i in range(n):
            result = highPrecision.sgp4Predict(line1, line2, jds[i], seconds[i])
            d_err.append(np.linalg.norm(result[0]-positions[i]))
            r_p.append(result[0])
        d_err = np.array(d_err)
        r_p = np.array(r_p)
        r_err = r_p - positions

    else:
        print("Reading Gravity Model ...")
        egmF = egm.readGFC(".\\egm\\JGM3.gfc")
        # get rs, vs, ts data
        rs, vs, times, init_jd = transform.getData(positions, times, jds, seconds)
        # use the first data as initial state
        init_elements = transform.states2Elements(rs[0], vs[0])
        # elements record
        order_used = 22
        lc = egm.LegendreCoef(order_used)
        result_per = predict.predict(init_elements, times, egmF, init_jd, order_used, lc)
        elements = np.transpose(result_per.get("y"))

        r_err = []
        d_err = []
        n = times.size
        for i in range(n):
            var = transform.getVar(elements[i])
            r_err.append(var.r - rs[i])
            d_err.append(np.linalg.norm(var.r - rs[i]))
        r_err = np.array(r_err)
        d_err = np.array(d_err)
        
    # draw result
    plt.subplot(121)
    plt.plot(times/3600, r_err[:, 0], label="x error")
    plt.plot(times/3600, r_err[:, 1], label="y error")
    plt.plot(times/3600, r_err[:, 2], label="z error")
    plt.legend()
    plt.grid()
    plt.xlabel("Time(h)")
    plt.ylabel("Error(m)")
    plt.title("Position errors of each dimension")

    plt.subplot(122)
    plt.plot(times/3600, d_err, color="orange")
    plt.title("Total position errors")
    plt.xlabel("Time(h)")
    plt.ylabel("Error(m)")
    plt.grid()

    plt.show()

# execution
main()
