######################################################
# Code to read profiles from input.gacode file #
######################################################

import numpy as np

######################################################


def single_int(f, name):
    datafile = open(name, "r")
    find = False
    for line in datafile:
        if find:
            data = int(line)
            break
        if f in line:
            find = True
    return data


######################################################


def single_float(f, name):
    datafile = open(name, "r")
    find = False
    for line in datafile:
        if find:
            data = float(line)
            break
        if f in line:
            find = True
    return data


######################################################


def mult_float(f, n, name):
    datafile = open(name, "r")
    find = False
    data = np.zeros([n])
    for line in datafile:
        if find:
            # d = []
            ii = 0
            for y in line.split():
                data[ii] = y
                ii += 1
            break
        if f in line:
            find = True
    return data


######################################################


def single_float_x(f, nx, name):
    datafile = open(name, "r")
    find = False
    data = np.zeros([2, nx])
    ii = 0
    for line in datafile:
        if find:
            for y in line.split():
                data[:, ii] = y
            ii += 1
        if ii == nx:
            break
        if f in line:
            find = True
    return data[1, :]


######################################################


def mult_float_x(f, nx, n, name):
    datafile = open(name, "r")
    find = False
    data = np.zeros([1 + n, nx])
    ii = 0
    for line in datafile:
        if find:
            for y in line.split():
                data[:, ii] = y
            ii += 1
        if ii == nx:
            break
        if f in line:
            find = True
    return data[1:, :]


######################################################


def miller_RZ(ny, data):

    nx = data["nexp"]
    R = np.zeros([nx, ny])
    Z = np.zeros([nx, ny])

    theta = np.linspace(0.0, 2.0 * np.pi, ny)
    for i in range(nx):
        R[i, :] = data["rmaj"][i] + data["rmin"][i] * np.cos(
            theta + data["delta"][i] * np.sin(theta)
        )
        Z[i, :] = data["kappa"][i] * data["rmin"][i] * np.sin(theta)

    return R, Z


######################################################


def var_2D(nx, ny, v):
    v2d = np.zeros([nx, ny])
    for i in range(nx):
        v2d[i, :] = v[i]
    return v2d


######################################################


def get_gacode_data(name):

    data = {}

    nexp = single_int("nexp", name)
    nion = single_int("nion", name)

    data["nexp"] = nexp
    data["nion"] = nion

    data["tor_flux"] = single_float("torfluxa | Wb/radian", name)
    data["B0"] = single_float("bcentr | T", name)
    data["R0"] = single_float("rcentr | m", name)
    data["I0"] = single_float("current | MA", name)

    data["mass_e"] = single_float("masse", name)
    data["mass_ion"] = mult_float("mass", nion, name)

    data["psi_1d"] = single_float_x("polflux | Wb/radian", nexp, name)
    data["psi_N"] = data["psi_1d"] / data["psi_1d"][-1]
    data["rho_pol"] = np.sqrt(data["psi_N"])

    data["rmin"] = single_float_x("rmin | m", nexp, name)
    data["rmaj"] = single_float_x("rmaj | m", nexp, name)

    data["q"] = single_float_x("q | -", nexp, name)
    data["kappa"] = single_float_x("kappa | -", nexp, name)
    data["delta"] = single_float_x("delta | -", nexp, name)

    data["z_e"] = single_float("ze", name)
    data["z_ion"] = mult_float("z", nion, name)

    data["n_e"] = single_float_x("ne | 10^19/m^3", nexp, name)
    data["n_ion"] = mult_float_x("ni | 10^19/m^3", nexp, nion, name)

    data["t_e"] = single_float_x("te | keV", nexp, name)
    data["t_ion"] = mult_float_x("ti | keV", nexp, nion, name)

    data["w0"] = single_float_x("w0 | rad/s", nexp, name)
    data["pressure"] = single_float_x("ptot | Pa", nexp, name)
    data["zeff"] = single_float_x("z_eff | -", nexp, name)

    ny = 57
    # nx = data["nexp"]

    R, Z = miller_RZ(ny, data)

    data["R"] = R
    data["Z"] = Z

    ii = 0
    points = np.zeros([(nexp - 1) * ny, 2])
    values = np.zeros([(nexp - 1) * ny])
    psi2D = np.zeros([nexp, ny])
    for i in range(1, nexp):
        for j in range(ny):
            points[ii, :] = [data["R"][i, j], data["Z"][i, j]]
            values[ii] = data["psi_1d"][i]
            psi2D[i, j] = data["psi_1d"][i]
            ii += 1

    data["psi2D"] = psi2D

    nx_int = 512
    ny_int = 512
    d = 0.5 * (np.max(R) + np.min(R)) * 0.05
    R_xy = np.linspace(np.min(R) - d, np.max(R) + d, nx_int)
    Z_xy = np.linspace(np.min(Z) - d, np.max(Z) + d, ny_int)
    x, y = np.meshgrid(R_xy, Z_xy)
    psi_xy = np.zeros([nx_int, ny_int])

    from scipy.interpolate import griddata

    psi_int = griddata(points, values, (x.T, y.T), method="cubic", fill_value=np.nan)

    for j in range(ny_int):
        psi_xy[:, j] = psi_int[j][:]

    data["R_xy"] = R_xy
    data["Z_xy"] = Z_xy
    data["psi_xy"] = psi_xy
    data["rho_xy"] = np.sqrt(psi_xy / data["psi_1d"][-1])

    return data


######################################################
######################################################

# name = 'input.gacode'
# data = get_gacode_data(name)
#
# ny = 57
# nx = data['nexp']
#
# R,Z = miller_RZ(ny,data)
#
# plt.figure(1)
# plt.plot(R,Z,color='black')
# plt.plot(R.T,Z.T,color='black')
# plt.axis('scaled')
#
# plt.show()
