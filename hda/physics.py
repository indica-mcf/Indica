"""
General equations and quantities for plasma physics
"""

import numpy as np
import scipy.constants as constants
from scipy.interpolate import interp1d
import math
from copy import deepcopy


def conductivity_spitzer(el_dens, el_temp, zeff, approx="sauter"):
    """
    Calculate Spitzer conductivity

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge
    approx
        Approximation chosen:
        "sauter" for:
            Sauter et al. Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240
        "todd" for:
            T. Todd et al.
            http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
            doc/TNTodd_Report_NeoclassicalResistivity2003.pdf


    Returns
    -------
    Spitzer conductivity (m**-1 ohm**-1)
    """

    if approx == "sauter":
        return conductivity_spitzer_sauter(el_dens, el_temp, zeff)
    elif approx == "todd":
        return conductivity_spitzer_todd(el_dens, el_temp, zeff)
    else:
        print(f"\n Approximation {approx} not implemented \n")


def conductivity_spitzer_sauter(el_dens, el_temp, zeff):
    """
    Calculate Spitzer conductivity following Sauter et al.
    Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge

    Returns
    -------
    Spitzer conductivity (m**-1 ohm**-1)

    """

    def n_zeff(zeff):
        return 0.58 + 0.74 / (0.76 + zeff)

    coul_log = coul_log_e(el_temp, el_dens)

    conductivity = 1.9012e4 * el_temp ** (3 / 2) / (zeff * n_zeff(zeff) * coul_log)

    return conductivity


def conductivity_spitzer_todd(el_dens, el_temp, zeff):
    """
    Calculate Spitzer conductivity following T. Todd et al.
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge

    Returns
    -------
    Spitzer conductivity (m**-1 ohm**-1)

    """

    def lambda_e_zeff(zeff):
        return 3.4 / zeff * (1.13 + zeff) / (2.67 + zeff)

    coul_log = coul_log_e(el_temp, el_dens)
    const = (
        3
        * (2 * math.pi) ** 1.5
        * constants.epsilon_0 ** 2
        / np.sqrt(constants.m_e * constants.e)
    )
    conductivity = const * el_temp ** (3 / 2) * lambda_e_zeff(zeff) / coul_log

    return conductivity


def conductivity_neo(
    el_dens, el_temp, zeff, min_r, a, R_mag, q, vloop=None, approx="sauter"
):
    """
    Calculate Spitzer conductivity

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge
    min_r
        minor radius (m)
    R_mag
        major radius (m)
    q
        safety factor
    vloop
        loop voltage (V)
    approx
        Approximation chosen:
        "sauter" for:
            Sauter et al. Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240
        "todd" for:
            T. Todd et al.
            http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
            doc/TNTodd_Report_NeoclassicalResistivity2003.pdf


    Returns
    -------
    Spitzer conductivity (m**-1 ohm**-1)
    """

    if approx == "sauter":
        return conductivity_neo_sauter(el_dens, el_temp, zeff, min_r, a, R_mag, q)
    elif approx == "todd" and vloop:
        return conductivity_neo_todd(el_dens, el_temp, zeff, min_r, a, R_mag, q, vloop)
    else:
        print(f"\n Approximation {approx} not implemented \n")


def conductivity_neo_sauter(el_dens, el_temp, zeff, min_r, a, R_mag, q):
    """
    Calculate neoclassical conductivity following Sauter et al.
    Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge
    a
        minor radius (m)
    R_mag
        major radius (m)
    q
        safety factor

    Returns
    -------
    Neoclassical conductivity (m**-1 ohm**-1)

    """

    def f33_teff(el_dens, el_temp, zeff, min_r, a, R_mag, q):
        ft = trapped_fraction(min_r, R_mag)
        nu_e = collisionality_electrons_sauter(el_dens, el_temp, zeff, a, R_mag, q)

        return ft / (
            1
            + (0.55 - 0.1 * ft) * np.sqrt(nu_e)
            + 0.45 * (1 - ft) * nu_e / zeff ** (3 / 2)
        )

    def F33(x):
        return (
            1 - (1 + 0.36 / zeff) * x + (0.59 / zeff) * x ** 2 - (0.23 / zeff) * x ** 3
        )

    spitzer_cond = conductivity_spitzer_sauter(el_dens, el_temp, zeff)

    x = f33_teff(el_dens, el_temp, zeff, min_r, a, R_mag, q)
    conductivity = spitzer_cond * F33(x)

    return conductivity


def conductivity_neo_todd(el_dens, el_temp, zeff, min_r, a, R_mag, q, vloop):
    """
    Calculate neoclassical conductivity following T. Todd et al.
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge
    a
        minor radius (m)
    R_mag
        major radius (m)
    q
        safety factor

    Returns
    -------
    Neoclassical conductivity (m**-1 ohm**-1)

    """

    spitzer_cond = conductivity_spitzer_todd(el_dens, el_temp, zeff)

    ft = trapped_fraction(min_r, R_mag)
    nu_e = collisionality_electrons_todd(el_dens, el_temp, a, R_mag, q, vloop)
    cr = 0.56 / zeff * (3.0 - zeff) / (3.0 - zeff)
    xi = 0.58 + 0.2 * zeff

    conductivity = (
        spitzer_cond * (1 - ft / (1 + xi * nu_e)) * (1 - cr * ft / (1 + xi * nu_e))
    )

    return conductivity


def trapped_fraction(min_r, R_mag):
    """
    Calculate fraction of trapped particles

    To be implemented following T. Todd
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf

    or Sauter et al.
    Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240

    Parameters
    ----------
    min_r
        minor radius (m)
    R_mag
        major radius (m)

    Returns
    -------
    Trapped particle fraction

    """
    eps = min_r / R_mag
    ft = np.sqrt(2 * eps / (1 + 2 * eps))

    return ft


def collisionality_electrons_sauter(el_dens, el_temp, zeff, a, R_mag, q):
    """
    Calculate collisionality parameter for electrons following Sauter et al.
    Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240)

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Electron temperature (eV)
    zeff
        Effective charge
    a
        minor radius
    R_mag
        major radius
    q
        safety factor profile

    Returns
    -------
    Electron collisionality

    """
    coul_log = coul_log_e(el_dens, el_temp)
    epsilon = a / R_mag
    nu = (
        6.921e-18
        * (q * R_mag * el_dens * zeff * coul_log)
        / (el_temp ** 2 * epsilon ** (3 / 2))
    )

    return nu


def collisionality_electrons_todd(el_dens, el_temp, a, R_mag, q, vloop):
    """
    Calculate collisionality parameter for electrons
    (see T. Todd
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf)

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Species temperature (eV)
    R_mag
        major radius (m)
    q
        safety factor profiles
    vloop
        loop voltage (V)

    Returns
    -------
    Electron collisionality

    """

    vthe = thermal_velocity(el_temp, ion=False)
    eps = a / R_mag
    E_perp = constants.m_e * vthe ** 2 / constants.e
    E_phi = 2 * math.pi * R_mag * vloop
    x = q * R_mag * E_phi / (eps * E_perp)
    tau_ee = collision_time_el_el(el_dens, el_temp)

    nu = (
        1.2
        * np.sqrt(2)
        * R_mag
        * q
        / vthe
        * eps ** (-3 / 2)
        * (1 - x ** 2) ** (-1 / 4)
        / tau_ee
    )
    return nu


def collision_time_el_el(el_dens, el_temp):
    """
    Calculate electron-electron collision time
    (see T. Todd
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf)

    Parameters
    ----------
    el_dens
        Electron density (m**-3)
    el_temp
        Species temperature (eV)

    Returns
    -------
    Electron-electron collision time (s)

    """

    vthe = thermal_velocity(el_temp, ion=False, ev=True)
    coul_log = coul_log_e(el_temp, el_dens)
    tau_ee = (
        3
        / (16 * np.sqrt(math.pi))
        * (4 * math.pi * constants.epsilon_0) ** 2
        * constants.m_e ** 2
        * vthe ** 3
        / (el_dens * constants.e ** 4 * coul_log)
    )

    return tau_ee


def collisionality_ions_sauter(ion_dens, ion_temp, charge, a, R_mag, q):
    """
    Calculate collisionality parameter for ions
    (see Sauter et al. Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240)

    Parameters
    ----------
    ion_dens
        Ion density (m**-3)
    ion_temp
        Ion temperature (eV)
    zeff
        Effective charge
    a
        minor radius
    R_mag
        major radius
    q
        safety factor profiles

    Returns
    -------
    Ion-ion collisionality

    """
    coul_log = coul_log_ii(ion_dens, ion_temp, charge)
    epsilon = a / R_mag
    nu = (
        4.9e-18
        * (q * R_mag * ion_dens * charge ** 4 * coul_log)
        / (ion_temp ** 2 * epsilon ** (3 / 2))
    )

    return nu


def coul_log_e(el_temp, el_dens):
    """
    Calculate electron Coulomb logarithm given temperature and density
    (see Sauter et al. Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240)

    Parameters
    ----------
    el_temp
        Electron temperature (eV)
    el_dens
        Electron density (m**-3)

    Returns
    -------
    Coulomb logarithm

    """
    return 31.3 - np.log(np.sqrt(el_dens) / el_temp)


def coul_log_ii(ion_temp, ion_dens, charge):
    """
    Calculate ion-ion Coulomb logarithm given temperature, density and charge
    (see Sauter et al. Physics of Plasmas 6, 2834 (1999); doi: 10.1063/1.873240)

    Parameters
    ----------
    ion_temp
        Ion temperature (eV)
    ion_dens
        Ion density (m**-3)
    charge
        Ion charge (units of electron charge)

    Returns
    -------
    Coulomb logarithm

    """
    return 30 - np.log(charge ** 3 * np.sqrt(ion_dens) / ion_temp ** (3 / 2))


def current_density(ipla, rho, a, area, prof_shape):
    """
    Build synthetic current density profile (A/m**2) given the
    total plasma current, plasma geometry and a shape parameter

    Parameters
    ----------
    ipla
        total plasma current (A)
    min_r
        minor radius (m)
    a
        separatrix minor radius (m)
    area
        poloidal cross-sectional aread
    nu
        shape parameter

    Returns
    -------

    """
    ir = np.where(rho >= 1)[0]

    j_phi = deepcopy(prof_shape)
    j_phi[ir] = 0.0

    j_phi = ipla * j_phi / np.trapz(j_phi, area)

    return j_phi


def resistance(resistivity, j_phi, area, ipla):
    """
    Calculate the plasma resistance given a current density and resistivity profile

    Parameters
    ----------
    resistivity
        Flux surface averaged resistivity profile (ohm)
    j_phi
        Flux surface averaged current density profile (A/m**2)
    area
        Poloidal cross-section area (m**2)
    ipla
        Current (A)

    Returns
    -------
        Plasma resistance

    """

    res = np.trapz(resistivity * j_phi ** 2, area) / ipla ** 2

    return res


def vloop(resistivity, j_phi, area):
    """
    Calculate the loop voltage given the current density, the resistivity
    and geometrical quantitie

    Parameters
    ----------
    resistivity
        Flux surface averaged resistivity profile (ohm)
    j_phi
        Flux surface averaged current density profile (A/m**2)
    area
        Poloidal cross-section area (m**2)
    ipla
        Current (A)

    Returns
    -------
        Plasma resistance

    """

    ipla = np.trapz(j_phi, area)
    vloop = ipla * resistance(resistivity, j_phi, area, ipla)

    return vloop


def poloidal_field(j_phi, min_r, area):
    """
    Calculate the poloidal field given current density and geometry

    Parameters
    ----------
    j_phi
        Toroidal current density (A/m**2)
    min_r
        minor radius (m)
    area
        poloidal cross-section area (m**2)

    Returns
    -------
        Poloidal field (T)

    """

    b_pol = np.zeros_like(j_phi)
    for ir in range(len(min_r)):
        if ir > 0:
            b_pol[ir] = (
                constants.mu_0
                * np.trapz(j_phi[:ir], area[:ir])
                / (2 * math.pi * min_r[ir])
            )

    b_pol[1] = b_pol[2] / 2.0

    return b_pol


def toroidal_field(bt_0, R_mag, maj_r):
    """
    Calculate the toroidal field given vacuum field

    Parameters
    ----------
    bt_0
        Toroidal magnetic field at reference radius R_mag (T)
    R_mag
        Reference major radius
    maj_r
        Major radius (m)

    Returns
    -------
        Toroidal field (T)

    """

    return bt_0 * R_mag / maj_r


def safety_factor(b_tor, b_pol, min_r, a, R_mag, monotonic=True):
    """
    TODO: Calculate safety factor profile following correct formula
    (see T. Todd
    http://mtc-m16.sid.inpe.br/col/sid.inpe.br/marciana/2004/01.15.15.07/
    doc/TNTodd_Report_NeoclassicalResistivity2003.pdf)

    !!!! CURRENTLY JUST APPROXIMATION !!!!

    Parameters
    ----------
    b_tor
        toroidal magnetic field (T)
    b_pol
        poloidal magnetic field (T)
    min_r
        minor radius
    R_mag
        major radius

    Returns
    -------
    safety factor profile

    """
    ir = np.where((min_r < a) * (b_pol > 0))[0]
    q_prof = np.full(min_r.size, np.nan)
    q_prof[ir] = (min_r[ir] * b_tor[ir]) / (R_mag * b_pol[ir])

    if monotonic:
        ir_interp = ir[
            np.where((np.gradient(q_prof[ir]) > 0) * (min_r[ir] > min_r[3]))[0]
        ]
        q_prof[0] = q_prof[np.min(ir_interp)] - 0.05
        ir_interp = np.concatenate([np.array([0]), ir_interp])
        finterp = interp1d(min_r[ir_interp], q_prof[ir_interp], kind="quadratic")
        q_prof[ir] = finterp(min_r[ir])

    return q_prof


def internal_inductance(b_pol, ipla, volume, approx=2, **kwargs):
    """
    Calculate the internal inductance given poloidal magnetic field and geometrical quantities

    Parameters
    ----------
    b_pol
        Poloidal magnetic field (T)
    ipla
        Total plasma current (A)
    approx
        Selected approximation

    kwargs
        min_r
            minor radius (m)
        a
            separatrix minor radius (m)
        R_geo
            geometric major radius
        R_mag
            magnetic major radius
        ipla
            total plasma current (A)
        volume
            plasma volume (m**3)
        elongation
            plasma elongation

    Returns
    -------
        Internal inductance l_i

    """

    def li1(b_pol, ipla, volume, min_r, a, R_geo, elongation):
        elongation_a = elongation[np.argmin(np.abs(min_r - a))]
        return (
            2
            * np.trapz(b_pol ** 2, volume)
            / (R_geo * (constants.mu_0 * ipla) ** 2)
            * (1 + elongation ** 2 / (2 * elongation_a))
        )

    def li2(b_pol, ipla, volume, R_mag):
        return 2 * np.trapz(b_pol ** 2, volume) / (R_mag * (constants.mu_0 * ipla) ** 2)

    def li3(b_pol, ipla, volume, R_geo):
        return 2 * np.trapz(b_pol ** 2, volume) / (R_geo * (constants.mu_0 * ipla) ** 2)

    if (
        approx == 1
        and "min_r" in kwargs.keys()
        and "a" in kwargs.keys()
        and "R_geo" in kwargs.keys()
        and "elongation" in kwargs.keys()
    ):
        return li1(
            b_pol,
            ipla,
            volume,
            kwargs["min_r"],
            kwargs["a"],
            kwargs["R_geo"],
            kwargs["elongation"],
        )

    if approx == 2 and "R_mag" in kwargs.keys():
        return li2(
            b_pol,
            ipla,
            volume,
            kwargs["R_mag"],
        )

    if approx == 3 and "R_geo" in kwargs.keys():
        return li3(
            b_pol,
            ipla,
            volume,
            kwargs["R_geo"],
        )

    return l_i


def beta(b_field, pressure, volume):
    """
    Calculate ratio of magnetic to kinetic pressure given magnetic field, pressure and geometrical quantities

    Parameters
    ----------
    b_field
        Magnetic field (T)
    pressure
        Plasma pressure (Pa)
    volume
        plasma volume

    Returns
    -------
        Beta

    """

    beta = (2 * constants.mu_0 * np.trapz(pressure, volume)) / np.trapz(
        b_field ** 2, volume
    )

    return beta


def beta_N():
    """
    Calculate normalised beta

    Parameters
    ----------

    Returns
    -------
        Normalised Beta

    """

    return None


def beta_poloidal(b_pol, pressure, volume):
    """
    Calculate the internal inductance given poloidal magnetic field and geometrical quantities

    Parameters
    ----------
    b_pol
        Poloidal magnetic field (T)
    pressure
        Plasma pressure (Pa)
    volume
        plasma volume

    Returns
    -------
        Poloidal Beta

    """

    beta_pol = (2 * constants.mu_0 * np.trapz(pressure, volume)) / np.trapz(
        b_pol ** 2, volume
    )

    return beta_pol


def calc_pressure(density, temperature):
    """
    Calculates pressure in Pa given product

    Parameters
    ----------
    density
        density in units of (m**-3)
    temperature
        temperature in units of (eV)

    Returns
    -------

    """

    return density * temperature * constants.e


def thermal_velocity(temperature, mass=1, ion=False, ev=False):
    """
    Calculates the thermal velocity given temperature (eV) and mass (AMU)

    Parameters
    ----------
    temperature
        Species temperature in (eV)
    mass
        Species mass (AMU)
    ion
        If False: species = electron;
        if True: species is an ion
    ev
        If True: temperature is kept in eV and not converted to K

    Returns
    -------
    Thermal velocity (m/s)

    """
    ev2k = constants.physical_constants["electron volt-kelvin relationship"][0]
    k_B = constants.k
    if ion is False:
        amu2kg = constants.m_e
    else:
        amu2kg = constants.m_p

    if ev is False:
        vth = np.sqrt(temperature * ev2k * k_B / (mass * amu2kg))
    else:
        vth = np.sqrt(temperature / (mass * amu2kg))

    return vth


def doppler_ev(sigma, centroid, mass: float, sigma_instr=0.0, fwhm=False):
    """
    Convert spectral line width to a temperature (eV)

    Parameters
    ----------
    sigma
        Gaussian sigma
    centroid
        Gaussian centroid
    mass
        Atomic number of the emitting element
    sigma_instr
        Instrument function sigma
    fwhm
        Set to true if input sigmas are given as Full Width at Half Maximum
        with fwhm = sigma * 2 * np.sqrt(2 * np.log(2))

    Returns
    -------
    temperature
        Temperature of the species in eV

    """

    sigma_thermal = np.sqrt(sigma ** 2 - sigma_instr ** 2)

    J2eV = constants.physical_constants["joule-electron volt relationship"][0]
    temperature = (
        mass
        * constants.m_p
        * J2eV
        * constants.c ** 2
        * (sigma_thermal / centroid) ** 2.0
    )

    if fwhm is True:
        temperature /= 8 * np.log(2)

    return temperature


def ev_doppler(temperature, mass: float, fwhm=False):
    """
    Convert spectral line width to a temperature (eV)

    Parameters
    ----------
    temperature
        Temperature of the ion species in eV
    mass
        Atomic number of the emitting element
    fwhm
        Set to true if input sigma is to given as a Full Width at Half Maximum
        with fwhm = sigma * 2 * np.sqrt(2 * np.log(2))

    Returns
    -------
    dl_l
        Relative width of the spectral line expressed as sigma/centroid

    """

    J2eV = constants.physical_constants["joule-electron volt relationship"][0]
    dl_l = np.sqrt(temperature / (mass * constants.m_p * J2eV * constants.c ** 2))

    if fwhm is True:
        dl_l *= 2 * np.sqrt(2 * np.log(2))

    return dl_l


def derivative(y, x):
    nlen = len(x)
    der = np.zeros(nlen)
    for i in range(nlen - 1):
        if i > 0:
            der[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])

    finterp = interp1d(x[1 : nlen - 1], q_prof[1 : nlen - 1], kind="quadratic")
    der = finterp(x)

    return der


def gaussian(x, A, B, x_0, w):
    """Build Gaussian with known parameters"""
    return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B
