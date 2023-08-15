import numpy as np
from scipy import constants
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
from indica.operators.slowingdown import simulate_finite_source
from indica.operators.slowingdown import simulate_slowingdown

AMU2KG = constants.m_p
EV2J = constants.e

LOCATION_HNBI = np.array([[0.33704, 0.93884, 0.0]])
DIRECTION_HNBI = np.array([[-0.704, -0.709, 0.0]])

LOCATION_RFX = np.array([[-0.341, -0.940, 0.0]])
DIRECTION_RFX = np.array([[0.704, 0.709, 0.0]])

RFX_DEFAULTS = {
    "element": "d",
    "energy": 25.0e3,
    "power": 0.5e6,
    "power_frac": (0.7, 0.1, 0.2),  # TODO: rename fractions
    "divergence": (14 * 1e-3, 14e-3),
    "width": (0.025, 0.025),
    "location": (-0.3446, -0.9387, 0.0),
    "direction": (0.707, 0.707, 0.0),
    "focus": 1.8,
}
HNBI_DEFAULTS = {
    "element": "d",
    "energy": 55.0e3,
    "power": 1.0e6,
    "power_frac": (0.6, 0.3, 0.1),
    "divergence": (14 * 1e-3, 14e-3),
    "width": (0.025, 0.025),
    "location": (-0.3446, -0.9387, 0.0),
    "direction": (0.707, 0.707, 0.0),
    "focus": 1.8,
}


class NeutralBeam(DiagnosticModel):
    def __init__(
        self,
        name: str,
        instrument_method: str = "get_neutral_beam",
        element: str = "d",
        energy: float = 55.0e3,
        power: float = 1.0e6,
        power_frac: LabeledArray = [0.6, 0.3, 0.1],
        divergence: LabeledArray = [14 * 1e-3, 14e-3],
        width: LabeledArray = [0.025, 0.025],
        focus: float = 1.8,
        n_beamlets: int = 10,
        n_mc: int = 10,
        **kwargs,
    ):
        """

        Parameters
        ----------
        name
            String identifier of the system
        instrument_method
            Corresponding method to read system's data from ST40 database
        element
            Beam gas
        energy
            Beam energy (V) (TODO: substitute with voltage?)
        power
            Beam power (W)
        power_frac
            Fraction of power in 1st, 2nd and 3rd energy
        n_beamlets
            Number of beamlets
        n_mc
            Number of MC-like samples
        TODO: substitute energy with voltage (engineering parameter)
        TODO: substitute power with current?
        TODO: n_beamlets to be incorporated in LOS-transform as different LOSs
              --> will substitute evaluate_rho in slowingdown.py
              --> rho2d interpolation performed once!
              --> this can become a version of VOS-implementation
        TODO: rename anum, znum -> atomic_mass, atomic_number

        """

        self.name = name
        self.instrument_method = instrument_method
        self.element = element
        self.anum_beam = ELEMENTS[element.lower()][1]
        self.znum_beam = ELEMENTS[element.lower()][0]
        self.power = power
        self.power_frac = np.array(power_frac)
        self.energy = energy
        self.energy_frac = energy / np.arange(1, np.size(power_frac) + 1)
        self.focus = focus
        self.width = np.array(width)
        self.divergence = np.array(divergence)
        self.n_beamlets = n_beamlets
        self.n_mc = n_mc

    def get_beam_parameters(self):
        beam_params = {
            "element": self.element,
            "anum_beam": self.anum_beam,
            "znum_beam": self.znum_beam,
            "power": self.power,
            "power_frac": self.power_frac,
            "energy": self.energy,
            "energy_frac": self.energy_frac,
            "focus": self.focus,
            "divergence": self.divergence,
            "width": self.width,
        }

        return beam_params

    def set_beam_parameters(self, parameters: dict):
        for k, v in parameters:
            setattr(self, k, v)

    def _build_bckc_dictionary(self):
        self.bckc = {
            "fast_density": self.fast_density,
            "fast_pressure_parallel": self.fast_pressure_parallel,
            "fast_pressure_perpendicular": self.fast_pressure_perpendicular,
        }

    def __call__(
        self,
        Ne: DataArray = None,
        Te: DataArray = None,
        equilibrium: Equilibrium = None,
        main_ion: str = None,
        t: LabeledArray = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        Ne
            Electron density profile (m**-3)
        Te
            Electron temperature profile (eV)
        equilibrium
            indica.equilibrium.Equilibrium object
        main_ion
            String identifier of plasma main ion
        t
            Desired time(s) of analysis
        kwargs
            ...
        Returns
        -------
            Dictionary with fast ion densities and pressure

        TODO: rename Rv, zv, rho_v --> R_equil, z_equil, rho_equil
        TODO: Indica native rho starts at 0 - check rho_profile below is self-consistent
        TODO: current implementation assumes pure plasma: expand to include impurities?
        TODO: fast pressure currently split equally between parallel and perpendicular
        """

        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate

            Ne = self.plasma.electron_density.interp(t=t, method="nearest")
            Te = self.plasma.electron_temperature.interp(t=t, method="nearest")
            equilibrium = self.plasma.equilibrium
            main_ion = self.plasma.main_ion
        else:
            if Ne is None or Te is None or equilibrium is None or main_ion is None:
                raise ValueError("Give inputs or assign plasma class!")

        R_equil = np.array(equilibrium.rho.R)
        z_equil = np.array(equilibrium.rho.z)
        rho_equil = np.array(equilibrium.rho.sel(t=t, method="nearest"))
        rho_profile = np.array(Ne.rho_poloidal)
        rho_profile[0] = rho_profile[1] / 2.0
        vol_profile = np.array(
            equilibrium.volume.sel(t=t, method="nearest")
            .diff("rho_poloidal")
            .interp(rho_poloidal=rho_profile)
        )

        width = self.width.mean()
        anum_plasma = ELEMENTS[main_ion][1]
        znum_plasma = ELEMENTS[main_ion][0]
        source = np.zeros((len(rho_profile), len(self.energy_frac)))

        for i in range(len(self.energy_frac)):
            source[:, i] = simulate_finite_source(
                rho_profile,
                Ne,
                Te,
                anum_plasma,
                R_equil,
                z_equil,
                rho_equil,
                vol_profile,
                self.los_transform.origin,
                self.los_transform.direction,
                self.energy_frac[i],
                self.anum_beam,
                self.power,
                width=width,
                n=self.n_beamlets,
            )

        result = simulate_slowingdown(
            Ne,
            Te,
            anum_plasma * AMU2KG,
            znum_plasma * EV2J,
            self.energy_frac,
            source,
            self.anum_beam * AMU2KG,
            self.znum_beam * EV2J,
            Nmc=self.n_mc,
        )

        self.Ne = Ne
        self.Te = Te
        self.equilibrium = equilibrium
        self.main_ion = main_ion
        self.t = t

        self.fast_density = result["nfast"]
        self.fast_pressure_parallel = result["pressure"] / 2.0
        self.fast_pressure_perpendicular = result["pressure"] / 2.0

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    pulse: int = None, t: float = None, n_beamlets: int = 10, n_mc: int = 10
):
    plasma = example_plasma(pulse=pulse)
    if t is None:
        t = plasma.t.mean()

    beam_name = "hnbi1"
    beam_params = HNBI_DEFAULTS
    beam_params["n_beamlets"] = n_beamlets
    beam_params["n_mc"] = n_mc

    origin = LOCATION_HNBI
    direction = DIRECTION_RFX
    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=beam_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    los_transform.set_equilibrium(plasma.equilibrium)
    model = NeutralBeam(beam_name, **beam_params)
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)
    bckc = model(t=t)

    return plasma, model, bckc
