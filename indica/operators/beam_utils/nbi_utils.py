"""Shared NBI data preparation utilities."""

import numpy as np

from indica.operators.beam_utils import adas_nbi_utils
from indica.operators.beam_utils import analytic_nbi_utils
from indica.operators.beam_utils import fidasim_utils


def get_model_handler(model_key: str):
    handlers = {
        "FIDASIM": fidasim_utils._run_fidasim,
        "ANALYTIC": analytic_nbi_utils._run_analytic,
        "ADAS": adas_nbi_utils._run_adas,
    }
    if model_key not in handlers:
        supported = ", ".join(handlers.keys())
        raise ValueError(
            f"Unknown nbi_model '{model_key}'. Supported models: {supported}"
        )
    return handlers[model_key]


def build_nbi_contexts(
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
    # Resolve missing inputs from the attached Plasma (if provided) or raise.
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

    # Validate required inputs are available.
    if (
        ion_temperature is None
        or electron_temperature is None
        or electron_density is None
        or neutral_density is None
        or toroidal_rotation is None
        or zeff is None
    ):
        raise ValueError("Give inputs or assign plasma class!")

    # Normalize time input into a 1D array.
    if t is None:
        t = ion_temperature.t
    t_values = np.atleast_1d(getattr(t, "data", t))

    # Resolve pulse and equilibrium from the transform.
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

    # Build a per-time context dict with profiles and equilibrium geometry.
    contexts = []
    for time in t_values:
        rho_1d = ion_temperature.rhop.values
        ion_temperature_t = ion_temperature.sel(t=time).values
        electron_temperature_t = electron_temperature.sel(t=time).values
        electron_density_t = electron_density.sel(t=time).values
        neutral_density_t = neutral_density.sel(t=time).values
        toroidal_rotation_t = toroidal_rotation.sel(t=time).values
        zeffective = zeff.sum("element").sel(t=time).values

        # rho poloidal
        rho_2d = eq.rhop.interp(
            t=time,
            method="nearest",
        )

        # rho toroidal
        rho_tor = eq.convert_flux_coords(rho_2d, t=time)
        rho_tor = rho_tor[0].values

        # radius
        R = eq.rhop.R.values
        z = eq.rhop.z.values
        R_2d, z_2d = np.meshgrid(R, z)

        # Br
        br, _ = eq.Br(
            eq.rhop.R,
            eq.rhop.z,
            t=time,
        )
        br = br.values

        # Bz
        bz, _ = eq.Bz(
            eq.rhop.R,
            eq.rhop.z,
            t=time,
        )
        bz = bz.values

        # Old
        irod = 3.0 * 1e6
        bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

        #New: sensible? Mostly nans in the test example so using Jonnys old estimate for now.
        bt2, _ = eq.Bt(
            eq.rhop.R,
            eq.rhop.z,
            t=time,
        )
        bt2 = bt2.transpose("z", "R").values


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
        contexts.append(ctx)

    return contexts
