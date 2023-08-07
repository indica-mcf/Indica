import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import scipy.interpolate as interp

epsilon0 = 8.854e-12
hbar = 1.054e-34


def coulomb_log(ma, qa, va, mb, qb, nb, Tb):
    """
    Calculate coulomb logarithm.

    ma: Test particle mass (kg)
    qa: Test particle charge (C)
    va: Test particle velocity (m/s)
    mb: Background species mass (kg)
    qb: Background species charge (C)
    nb: Background species density (1/m3)
    Tb: Background species temperature (1/m3)
    """
    debyeLength = np.sqrt(epsilon0 / np.sum(nb * qb * qb / Tb))
    vbar = va * va + 2 * Tb / mb
    mr = ma * mb / (ma + mb)
    bcl = np.abs(qa * qb / (4 * np.pi * epsilon0 * mr * vbar))
    bqm = np.abs(hbar / (2 * mr * np.sqrt(vbar)))

    clog = np.zeros(len(mb))

    for i in range(len(clog)):
        if bcl[i] > bqm[i]:
            clog[i] = np.log(debyeLength / bcl[i])
        else:
            clog[i] = np.log(debyeLength / bqm[i])

    return clog


def slowingdown(ma, qa, Ea, mb, qb, nb, Tb, Nmc=10, dt=1):
    """
    Calculate slowing-down for a fast particle. Collision operator adopted from 
    Hirvijoki et al 2014 CPC.

    ma: Fast particle mass (kg)
    qa: Fast particle charge (C)
    Ea: Fast particle energy (J)
    mb: Background species mass (kg)
    qb: Background species charge (C)
    nb: Background species density (1/m3)
    Tb: Background species temperature (eV)
    dt: Timestep for output (s)
    """
    max_tslow = 0.1

    tv = np.arange(0, max_tslow, dt)
    nfast = np.zeros(len(tv))
    pressure = np.zeros(len(tv))

    for i in range(Nmc):
        va = np.sqrt(2 * Ea / ma)
        vb = np.sqrt(2 * Tb / mb)

        t = 0
        dt = 1e-3

        while va > np.sqrt(2 * Tb / ma):
            t += dt
            mu0 = (
                erf(va / vb) - 2 * va / vb * np.exp(-va / vb * va / vb) / np.sqrt(np.pi)
            ) / (va / vb * va / vb)
            mu1 = erf(va / vb) - 0.5 * mu0
            clog = coulomb_log(ma, qa, va, mb, qb, nb, Tb)
            Cb = nb * qa * qa * qb * qb * clog / (4 * np.pi * epsilon0 * epsilon0)
            F = np.sum(-(1 / mb + 1 / ma) * Cb * mu0 / (ma * vb * vb))
            Dpara = np.sum(Cb * mu0 / (2 * ma * ma * va))
            Dperp = np.sum(Cb * mu1 / (2 * ma * ma * va))

            dW = np.random.normal()

            vav = np.array([va, 0, 0])
            vav[0] += F * dt + np.sqrt(2 * Dpara * dt) * dW
            vav[1] += np.sqrt(2 * Dperp * dt) * dW
            vav[2] += np.sqrt(2 * Dperp * dt) * dW
            va = np.linalg.norm(vav)

            it = np.argmin(np.abs(tv - t))

            nfast[it] += dt
            pressure[it] += ma * va * va / 3 * dt

    return nfast / Nmc, pressure / Nmc


def simulate_slowingdown(
    ne, Te, mass, charge, E_fast, source_fast, mass_fast, charge_fast, Nmc=10
):
    """
    Calculate steady-state (time independent) fast ion density and pressure. Assumes zero orbit
    width and no FLR.

    ne: Background electron density profile (1/m3) - profile on rho
    Te: Background electron temperature profile (eV)
    mass: Background species mass (kg) - main ion
    charge: Background species charge (C) - main ion
    E_fast: Fast ion energy (eV) - vectors of energy components [E, E/2, E/3]
    source_fast: Fast ion source profile (1/s m3) - (rho, source) source of beam components [E, E/2, E/3]
    mass_fast: Fast ion mass (kg) - main ion
    charge_fast: Fast ion charge (C) - main ion

    Returns:
    nfast: Fast ion density profile (1/m3) - profile on same radial grid as the sources in input
    pressure: Fast ion pressure profile (Pa)
    """
    N = len(ne)

    nfast = np.zeros(N)
    pressure = np.zeros(N)

    for i in range(N):
        for j in range(3):
            nfast_frac, pressure_frac = slowingdown(
                mass_fast,
                charge_fast,
                E_fast[j] * 1.602e-19,
                np.array([9.109e-31, mass]),
                np.array([-1.602e-19, charge]),
                ne[i],
                Te[i] * 1.602e-19,
                Nmc,
            )
            nfast[i] += source_fast[i, j] * nfast_frac
            pressure[i] += source_fast[i, j] * pressure_frac

    return nfast, pressure


def simulate_slowingdown_timedep(
    tv, ne, Te, mass, charge, E_fast, source_fast, mass_fast, charge_fast, Nmc=10
):
    """
    Calculate steady-state fast ion density and pressure. Assumes zero orbit
    width and no FLR.

    ne: Background electron density profile (1/m3)
    Te: Background electron temperature profile (eV)
    mass: Background species mass (kg)
    charge: Background species charge (C)
    E_fast: Fast ion energy (eV)
    source_fast: Fast ion source profile (1/s m3)
    mass_fast: Fast ion mass (kg)
    charge_fast: Fast ion charge (C)

    Returns:
    nfast: Fast ion density profile (1/m3)
    pressure: Fast ion pressure profile (Pa)
    """
    N = len(ne[0, :])
    Nt = len(tv)

    dt = tv[1] - tv[0]

    nfast = np.zeros((Nt, N))
    pressure = np.zeros((Nt, N))

    for it in range(Nt):
        for i in range(N):
            for j in range(3):
                nfast_frac, pressure_frac = slowingdown(
                    mass_fast,
                    charge_fast,
                    E_fast[j] * 1.602e-19,
                    np.array([9.109e-31, mass]),
                    np.array([-1.602e-19, charge]),
                    ne[it, i],
                    Te[it, i] * 1.602e-19,
                    Nmc,
                    dt,
                )

                nfast_frac = nfast_frac[0 : Nt - it]
                nfast[it : it + len(nfast_frac), i] += (
                    source_fast[it, i, j] * nfast_frac
                )

                pressure_frac = pressure_frac[0 : Nt - it]
                pressure[it : it + len(pressure_frac), i] += (
                    source_fast[it, i, j] * pressure_frac
                )

    return nfast, pressure


def suzuki(E, ne, Te, Anum):
    """
    Calculate beam-stopping coefficient from Suzuki et al.

    E: Neutral energy/amu (keV)
    ne: Electron density (1/m3)
    Te: Electron temperature (eV)
    """
    A = np.array(
        [
            [
                -52.9,
                -1.36,
                0.0719,
                0.0137,
                0.454,
                0.403,
                -0.22,
                0.0666,
                -0.0677,
                -0.00148,
            ],
            [
                -67.9,
                -1.22,
                0.0814,
                0.0139,
                0.454,
                0.465,
                -0.273,
                0.0751,
                -0.063,
                -0.000508,
            ],
            [
                -74.2,
                -1.18,
                0.0843,
                0.0139,
                0.453,
                0.491,
                -0.294,
                0.0788,
                -0.0612,
                -0.000185,
            ],
        ]
    )

    sigmav = (
        ne
        * A[Anum - 1, 0]
        * 1.0e-16
        / E
        * (1 + A[Anum - 1, 1] * np.log(E) + A[Anum - 1, 2] * np.log(E) * np.log(E))
        * (
            1
            + (1 - np.exp(-A[Anum - 1, 3] * ne * 1e-19)) ** A[Anum - 1, 4]
            * (
                A[Anum - 1, 5]
                + A[Anum - 1, 6] * np.log(E)
                + A[Anum - 1, 7] * np.log(E) * np.log(E)
            )
        )
        * (
            1
            + A[Anum - 1, 8] * np.log(Te * 1e-3)
            + A[Anum - 1, 9] * np.log(Te * 1e-3) * np.log(Te * 1e-3)
        )
    ) / ne

    return sigmav


def simulate_source(
    rhov, ne, Te, Anum, Rv, zv, rho2d, vol, source, direction, energy_per_anum
):
    """
    rhov - profiles rho values
    ne (m**-3) - profiles
    Te (eV) - profiles
    Anum - mass number main ion
    Rv - rho2d major radius grid
    zv - rho2d z grid
    rho2d - equilibrium rho 2D (R, z)
    vol (m**3) - equilibrium_data["vjac"] * dpsin, not cumulative sum!
    source (m) - position of the source in (x, y, z)
    direction (m) - position of the source in (x, y, z)
    energy_per_anum (keV) - 1 energy only, call this three times for each energy

    Returns
    -------

    """
    ds = 0.001
    s = 0
    x = source
    rho = evaluate_rho(x, Rv, zv, rho2d)

    # step until in plasma
    while rho > 1:
        x += ds * direction
        rho = evaluate_rho(x, Rv, zv, rho2d)

    source_fast = np.zeros(len(ne))
    weight = 1

    attenuation = []

    while rho <= 1:
        x += ds * direction
        s += ds
        rho = evaluate_rho(x, Rv, zv, rho2d)
        irho = np.argmin(np.abs(rho - rhov))
        rate = ne[irho] * 1e-4 * suzuki(energy_per_anum, ne[irho], Te[irho], Anum)
        source_fast[irho] += weight * rate * ds
        weight -= weight * rate * ds
        attenuation.append([s, weight])

    return source_fast / vol, np.array(attenuation)


def evaluate_rho(x, Rv, zv, rho2d):
    frho = interp.interp2d(Rv, zv, rho2d, kind="cubic")
    return frho(np.sqrt(x[0] ** 2 + x[1] ** 2), x[2])[0]
