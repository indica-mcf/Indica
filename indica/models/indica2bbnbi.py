import subprocess

from a5py.ascot5io import dist_6D as d6d
import a5py.ascot5io.B_2DS as B_2D
import a5py.ascot5io.E_TC as E_TC
import a5py.ascot5io.N0_3D as N0_3D
import a5py.ascot5io.plasma_1D as P_1D
import a5py.ascot5io.wall_2D as W_2D
import a5py.nbi.generate_injector as generate_injector
import numpy as np
import scipy.interpolate as interp

# temporary file name for bbnbi run
fn = "temp.h5"


def indica2bbnbi(time, equil, el_dens, el_temp, ion_dens, ion_temp, beam):
    """ """

    # create 2D magetic field input
    B2D = dict()
    it = np.argmin(np.abs(np.array(equil.psi["t"]) - time))
    B2D["Rmin"] = equil.psi["R"][0]
    B2D["Rmax"] = equil.psi["R"][-1]
    B2D["nR"] = np.size(equil.psi["R"])
    B2D["zmin"] = equil.psi["z"][0]
    B2D["zmax"] = equil.psi["z"][-1]
    B2D["nz"] = np.size(equil.psi["z"])
    B2D["axisR"] = equil.rmag[it]
    B2D["axisz"] = equil.zmag[it]
    B2D["psiRz"] = np.array(equil.psi[it, :, :])
    fpsi = interp.interp2d(
        equil.psi["R"], equil.psi["z"], equil.psi[it, :, :], kind="cubic"
    )
    B2D["psiaxis"] = fpsi(equil.rmag[it], equil.zmag[it])
    B2D["psisepx"] = fpsi(equil.rbnd[it, 0], equil.zbnd[it, 0])
    psiv = equil.f["rho_poloidal"] * (B2D["psisepx"] - B2D["psiaxis"]) + B2D["psiaxis"]
    f = interp.interp1d(
        psiv,
        equil.f[it, :],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    Rgrid = np.tile(equil.psi["R"], [np.size(equil.psi["z"]), 1])
    B2D["B_R"] = 0 * B2D["psiRz"]
    B2D["B_z"] = 0 * B2D["psiRz"]
    B2D["B_phi"] = f(B2D["psiRz"]) / Rgrid
    B2D["B_phi"] = B2D["B_phi"].T
    B_2D.write_hdf5(
        fn,
        B2D["Rmin"],
        B2D["Rmax"],
        B2D["nR"],
        B2D["zmin"],
        B2D["zmax"],
        B2D["nz"],
        B2D["axisR"],
        B2D["axisz"],
        B2D["psiRz"].T,
        B2D["psiaxis"],
        B2D["psisepx"],
        B2D["B_R"],
        B2D["B_phi"],
        B2D["B_z"],
    )

    # create 1D plasma input
    pls = dict()
    it = np.argmin(np.abs(np.array(el_dens["t"]) - time))
    pls["rho"] = el_dens["rho_poloidal"]
    pls["nrho"] = len(pls["rho"])
    pls["nion"] = 1
    pls["anum"] = 1
    pls["mass"] = 1.007
    pls["znum"] = 1
    pls["charge"] = 1
    pls["edensity"] = el_dens[it, :]
    pls["idensity"] = np.array([el_dens[it, :]]).T
    pls["etemperature"] = el_temp[it, :]
    pls["itemperature"] = ion_temp[0, it, :]
    P_1D.write_hdf5(fn, **pls)

    # create beam geometry input
    beam_loc = np.array(beam["location"])
    beam_dir = np.array(beam["direction"])
    beam_R = np.sqrt(beam["location"][0] ** 2 + beam["location"][1] ** 2)
    beam_phi = np.arctan2(beam["location"][1], beam["location"][0]) * 180.0 / np.pi
    beam_tanrad = np.linalg.norm(-beam_loc + np.dot(beam_loc, beam_dir) * beam_dir)
    generate_injector.generate(
        fn,
        beam_R,
        beam_phi,
        beam["location"][2],
        beam_tanrad,
        beam["focus"],
        beam["width"],
        50000,
        beam["amu"],
        1,
        beam["amu"],
        beam["energy"],
        beam["fractions"][0:3],
        beam["power"],
        beam["divergence"],
    )

    # add dummy inputs not used in bbnbi run
    N0_3D.write_hdf5_dummy(fn)
    E_TC.write_hdf5_dummy(fn)
    W_2D.write_hdf5_dummy(fn)


def run(n_markers=100000):
    """ """
    subprocess.run(
        [
            "/home/jari.varje/git/ascot5_bbnbimod/bbnbi5",
            "--in",
            fn[:-3],
            "--out",
            fn[:-3],
            "--n",
            str(n_markers),
        ]
    )


def bbnbi2indica():
    """ """
    dist = d6d.read_hdf5(fn, "1234567890")
    hist = np.squeeze(dist["histogram"])

    for i in range(len(dist["r"])):
        for j in range(len(dist["phi"])):
            for k in range(len(dist["z"])):
                hist[i, j, k] /= (
                    2
                    * np.pi
                    * (dist["r_edges"][i] + dist["r_edges"][i + 1])
                    / 2
                    * (dist["r"][1] - dist["r"][0])
                    * (dist["z"][1] - dist["z"][0])
                )

    dims = (dist["r"], dist["phi"], dist["z"])
    return hist, dims
