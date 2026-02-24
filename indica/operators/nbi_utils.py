"""Shared NBI data preparation utilities."""

import numpy as np


def build_nbi_context(
    operator,
    ion_temperature=None,
    electron_temperature=None,
    electron_density=None,
    neutral_density=None,
    toroidal_rotation=None,
    zeff=None,
    t=None,
    pulse: int = None,
    plasma=None,
):
    if plasma is not None:
        operator.plasma = plasma
    if operator.plasma is not None:
        if ion_temperature is None:
            ion_temperature = operator.plasma.ion_temperature
        if electron_temperature is None:
            electron_temperature = operator.plasma.electron_temperature
        if electron_density is None:
            electron_density = operator.plasma.electron_density
        if neutral_density is None:
            neutral_density = operator.plasma.neutral_density
        if toroidal_rotation is None:
            toroidal_rotation = operator.plasma.toroidal_rotation
        if zeff is None:
            zeff = operator.plasma.zeff
        if t is None:
            t = getattr(operator.plasma, "time_to_calculate", None)
            if t is None:
                t = operator.plasma.t

    if (
        ion_temperature is None
        or electron_temperature is None
        or electron_density is None
        or neutral_density is None
        or toroidal_rotation is None
        or zeff is None
    ):
        raise ValueError("Give inputs or assign plasma class!")

    if t is None:
        t = ion_temperature.t
    t_values = np.atleast_1d(getattr(t, "data", t))
    if t_values.size != 1:
        raise ValueError("Expected a single time value for NBI model run.")

    if pulse is None:
        pulse = operator.pulse
    if pulse is None:
        raise ValueError("pulse is required (set it on init or pass to __call__)")

    if operator.transform is None:
        raise ValueError("transform is required (set it before calling)")
    if (
        not hasattr(operator.transform, "equilibrium")
        or operator.transform.equilibrium is None
    ):
        raise ValueError("transform is missing equilibrium data")
    eq = operator.transform.equilibrium

    profiles = {
        "t": t_values,
        "ion_temperature": ion_temperature,
        "electron_temperature": electron_temperature,
        "electron_density": electron_density,
        "neutral_density": neutral_density,
        "toroidal_rotation": toroidal_rotation,
        "zeff": zeff,
    }
    eqdata = {
        "rhop": eq.rhop,
        "convert_flux_coords": eq.convert_flux_coords,
        "Br": eq.Br,
        "Bz": eq.Bz,
    }

    time = profiles["t"][0]
    rho_1d = profiles["ion_temperature"].rhop.values
    ion_temperature_t = profiles["ion_temperature"].sel(t=time).values
    electron_temperature_t = profiles["electron_temperature"].sel(t=time).values
    electron_density_t = profiles["electron_density"].sel(t=time).values
    neutral_density_t = profiles["neutral_density"].sel(t=time).values
    toroidal_rotation_t = profiles["toroidal_rotation"].sel(t=time).values
    zeffective = profiles["zeff"].sum("element").sel(t=time).values

    # rho poloidal
    rho_2d = eqdata["rhop"].interp(
        t=time,
        method="nearest",
    )

    # rho toroidal
    rho_tor = eqdata["convert_flux_coords"](rho_2d, t=time)
    rho_tor = rho_tor[0].values

    # radius
    R = eqdata["rhop"].R.values
    z = eqdata["rhop"].z.values
    R_2d, z_2d = np.meshgrid(R, z)

    # Br
    br, _ = eqdata["Br"](
        eqdata["rhop"].R,
        eqdata["rhop"].z,
        t=time,
    )
    br = br.values

    # Bz
    bz, _ = eqdata["Bz"](
        eqdata["rhop"].R,
        eqdata["rhop"].z,
        t=time,
    )
    bz = bz.values

    # Bt (toroidal field estimate)
    irod = 3.0 * 1e6
    bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

    rho = rho_2d.values
    ctx = {
        "pulse": pulse,
        "time": time,
        "rho_1d": rho_1d,
        "rho": rho,
        "rho_tor": rho_tor,
        "R_2d": R_2d,
        "z_2d": z_2d,
        "br": br,
        "bz": bz,
        "bt": bt,
        "ion_temperature": ion_temperature_t,
        "electron_temperature": electron_temperature_t,
        "electron_density": electron_density_t,
        "neutral_density": neutral_density_t,
        "toroidal_rotation": toroidal_rotation_t,
        "zeffective": zeffective,
    }

    return ctx
