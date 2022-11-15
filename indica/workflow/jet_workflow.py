"""
Base script for running InDiCA analysis tests.
Run with specific data source (e.g. JET JPF/PPF data)
"""
from copy import deepcopy
import getpass
from pathlib import Path
from socket import getfqdn
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from sal.core.exception import NodeNotFound
import xarray as xr
from xarray import concat
from xarray import DataArray

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.operators import BolometryDerivation
from indica.operators import InvertRadiation
from indica.operators import SplineFit
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity
from indica.operators.extrapolate_impurity_density import recover_threshold_rho
from indica.operators.impurity_concentration import ImpurityConcentration
from indica.operators.invert_radiation import EmissivityProfile
from indica.operators.main_ion_density import MainIonDensity
from indica.operators.mean_charge import MeanCharge
from indica.readers import ADASReader
from indica.readers import PPFReader
from indica.readers.selectors import ignore_channels_from_dict
from indica.utilities import to_filename
from .base_workflow import BaseWorkflow
from .workflow_utilities import bolo_los
from .workflow_utilities import print_step_template


class JetWorkflow(BaseWorkflow):
    """
    Setup and run standard analysis for benchmarking InDiCA against WSX on JET
    data, where JET data readers are available
    """

    _initial_sxr_calibration_factor: float = 2.8

    def __init__(
        self,
        config: Dict[str, Any] = None,
        config_file: Union[str, Path] = "input.json",
    ):
        super().__init__(config=config, config_file=config_file)
        self.pulse: int = int(self.input.get("pulse", 0))
        self.cache_file = self.cache_dir / f"{self.pulse}.json"
        # abstractreader.CACHE_DIR = str(self.cache_dir)
        try:
            self.reader = PPFReader(
                pulse=self.pulse,
                tstart=self.trange[0],
                tend=self.trange[1],
                server="https://sal.jetdata.eu"
                if "jetdata" in getfqdn().lower()
                else "https://sal.jet.uk",
                selector=ignore_channels_from_dict(
                    ignore_dict=self.input.get("ignore_channels", {}),
                    ignore_bad_channels=True,
                    use_cached_ignore=False,
                ),
            )
        except ConnectionError as e:
            raise ConnectionError(e)
        self.authenticate_reader()
        self.additional_data: Dict[str, Any] = {}

    def __call__(
        self,
        setup_only: bool = False,
        n_loops: int = 3,
        *args,
        **kwargs,
    ):
        super().__call__(*args, **kwargs)
        if setup_only is True:
            return
        template = print_step_template(fallback_size=(80, 24))
        # Analysis steps
        # Set initial densities to 0, using coords from sxr_emissivity
        self.initialise_default_values()
        for i in range(n_loops):
            print(template.format("calculate_n_high_z"))
            self.calculate_n_high_z()
        return self.n_high_z

    @property
    def __external_properties__(self) -> List[str]:
        """
        List of properties that are externally obtained
        """
        base = super().__external_properties__
        return [
            *base,
            *[
                "diagnostics",
                "efit_equilibrium",
                "flux_surface",
                "SCD",
                "ACD",
                "FA",
                "PLT",
                "PRB",
                "PL",
                "SXRPLT",
                "SXRPRB",
                "SXRPL",
                "sxr_fitted_symmetric_emissivity",
                "sxr_fitted_asymmetry_parameter",
                "sxr_emissivity",
                "electron_density",
                "electron_temperature",
                "ion_temperature",
                "toroidal_rotation",
                "cxrs_concentration",
                "fzt",
                "power_loss",
                "sxr_power_loss",
                "q",
                "additional_data",
            ],
        ]

    @property
    def setup_steps(self) -> List[Tuple[Callable, str]]:
        return [
            (self.get_diagnostics, "diagnostics"),
            (self.fit_diagnostic_profiles, "toroidal_rotation"),
            (self.invert_sxr, "sxr_emissivity"),
            (self.get_adas, "SXRPL"),
            (self.calculate_power_loss, "q"),
        ]

    def clean_cache(self):
        sal_cache = self.cache_dir / f"{self.reader.__class__.__name__}"
        for ppf_cache in sal_cache.glob(
            f"{to_filename(self.reader._reader_cache_id)}*"
        ):
            ppf_cache.unlink()
        return super().clean_cache()

    def authenticate_reader(self) -> None:
        if self.reader.requires_authentication:
            user = input("JET username: ")
            password = getpass.getpass("JET password: ")
            assert self.reader.authenticate(user, password)

    def get_diagnostics(self) -> None:
        """
        Get dictionary of relevant diagnostics and equilibrium object

        Returns
        -------
        """
        self.diagnostics = {
            "efit": self.reader.get(uid="jetppf", instrument="eftp", revision=0),
            "hrts": self.reader.get(uid="jetppf", instrument="hrts", revision=0),
            "sxr": self.reader.get(uid="jetppf", instrument="sxr", revision=0),
            "zeff": self.reader.get(uid="jetppf", instrument="ks3", revision=0),
            "bolo": self.reader.get(uid="jetppf", instrument="bolo", revision=0),
            "cxrs": self.reader.get(
                uid="jetppf",
                instrument=self.cxrs_instrument,
                revision=0,
                quantities={"ti", "angf"},
            ),
        }
        try:
            self.diagnostics["cxrs"]["conc"] = self.reader.get(
                uid="jetppf",
                instrument=self.cxrs_instrument,
                revision=0,
                quantities={"conc"},
            )["conc"]
        except (NodeNotFound, KeyError):
            pass
        self.efit_equilibrium = Equilibrium(equilibrium_data=self.diagnostics["efit"])
        for key, diag in self.diagnostics.items():
            for data in diag.values():
                if hasattr(data.attrs["transform"], "equilibrium"):
                    del data.attrs["transform"].equilibrium
                if "efit" not in key.lower():
                    data.indica.equilibrium = self.efit_equilibrium
        self.flux_surface = FluxSurfaceCoordinates(kind="poloidal")
        self.flux_surface.set_equilibrium(self.efit_equilibrium)

    def get_adas(self):
        adas = ADASReader()
        impurities = [val for val in self.ion_species if val != self.main_ion]
        self.SCD = {
            element: adas.get_adf11("scd", element, year)
            for element, year in zip(impurities, ["89"] * len(impurities))
        }
        self.SCD[self.main_ion] = adas.get_adf11("scd", "h", "89")
        self.ACD = {
            element: adas.get_adf11("acd", element, year)
            for element, year in zip(impurities, ["89"] * len(impurities))
        }
        self.ACD[self.main_ion] = adas.get_adf11("acd", "h", "89")
        self.FA = {
            element: FractionalAbundance(
                SCD=self.SCD.get(element), ACD=self.ACD.get(element)
            )
            for element in self.ion_species
        }
        self.PLT = {
            element: adas.get_adf11("plt", element, year)
            for element, year in zip(impurities, ["89"] * len(impurities))
        }
        self.PLT[self.main_ion] = adas.get_adf11("plt", "h", "89")
        self.PRB = {
            element: adas.get_adf11("prb", element, year)
            for element, year in zip(impurities, ["89"] * len(impurities))
        }
        self.PRB[self.main_ion] = adas.get_adf11("prb", "h", "89")
        self.PL = {
            element: PowerLoss(PLT=self.PLT.get(element), PRB=self.PRB.get(element))
            for element in self.ion_species
        }

        adas = ADASReader(
            self.input.get(
                "sxr_filtered_adas",
                "/home/elitherl/Analysis/SXR/indica/sxr_filtered_adf11/",
            )
        )
        self.SXRPLT = {
            element: adas.get_adf11("pls", element, year)
            for element, year in zip(impurities, ["5"] * len(impurities))
        }
        self.SXRPLT[self.main_ion] = adas.get_adf11("pls", "h", "5")
        self.SXRPRB = {
            element: adas.get_adf11("prs", element, year)
            for element, year in zip(impurities, ["5"] * len(impurities))
        }
        self.SXRPRB[self.main_ion] = adas.get_adf11("prs", "h", "5")
        self.SXRPL = {
            element: PowerLoss(
                PLT=self.SXRPLT.get(element), PRB=self.SXRPRB.get(element)
            )
            for element in self.ion_species
        }

    def invert_sxr(self):
        cameras: List[str] = self.input.get("cameras", ["v"])
        n_knots: int = self.input.get("n_knots", 7)
        inverter = InvertRadiation(
            num_cameras=len(cameras), datatype="sxr", n_knots=n_knots
        )
        emissivity, emiss_fit, *camera_results = inverter(
            self.R,
            self.z,
            self.t,
            *[self.diagnostics["sxr"][key] for key in cameras],
        )
        self.sxr_fitted_symmetric_emissivity = emiss_fit.symmetric_emissivity
        self.sxr_fitted_asymmetry_parameter = emiss_fit.asymmetry_parameter
        # TEMP whilst discussing InvertRadiation changes for new coordinate schemes
        # transform = FluxMajorRadCoordinates(self.flux_surface)
        transform = self.flux_surface
        emiss_profile = EmissivityProfile(
            self.sxr_fitted_symmetric_emissivity,
            self.sxr_fitted_asymmetry_parameter,
            self.flux_surface,
        )
        self.sxr_emissivity = (
            emiss_profile(transform, self.rho, self.theta, self.t).drop("z").drop("r")
        )
        self.additional_data["invert_sxr"] = {
            "cameras": cameras,
            "n_knots": n_knots,
            "inverter": inverter,
            "emiss_fit": emiss_fit,
            "camera_results": camera_results,
            "FluxMajorRadTransform": transform,
        }

    def fit_diagnostic_profiles(self):
        default_knots = [0.0, 0.3, 0.6, 0.85, 0.9, 0.98, 1.0, 1.05]
        knots_te = self.input.get("spline_knots", {}).get(
            "electron temperature", default_knots
        )
        fitter_te = SplineFit(
            lower_bound=0.0,
            upper_bound=self.diagnostics["hrts"]["te"].max() * 1.1,
            knots=knots_te,
        )
        results_te = fitter_te(self.rho, self.t, self.diagnostics["hrts"]["te"])
        self.electron_temperature = results_te[0]

        temp_ne = deepcopy(self.diagnostics["hrts"]["ne"])
        temp_ne.attrs["datatype"] = deepcopy(
            self.diagnostics["hrts"]["te"].attrs["datatype"]
        )  # TEMP for SplineFit checks
        knots_ne = self.input.get("spline_knots", {}).get(
            "electron density", default_knots
        )
        fitter_ne = SplineFit(
            lower_bound=0.0, upper_bound=temp_ne.max() * 1.1, knots=knots_ne
        )
        results_ne = fitter_ne(self.rho, self.t, temp_ne)
        self.electron_density = results_ne[0]

        temp_ti = deepcopy(self.diagnostics["cxrs"]["ti"])
        temp_ti.attrs["datatype"] = deepcopy(
            self.diagnostics["hrts"]["te"].attrs["datatype"]
        )  # TEMP for SplineFit checks
        temp_angf = deepcopy(self.diagnostics["cxrs"]["angf"])
        temp_angf.attrs["datatype"] = deepcopy(
            self.diagnostics["hrts"]["te"].attrs["datatype"]
        )  # TEMP for SplineFit checks
        knots_ti = self.input.get("spline_knots", {}).get(
            "ion temperature", default_knots
        )
        fitter_ti = SplineFit(
            lower_bound=temp_ti.min() * 0.9,
            upper_bound=temp_ti.max() * 1.1,
            knots=knots_ti,
        )
        results_ti = fitter_ti(self.rho, self.t, temp_ti)
        self.ion_temperature = results_ti[0]
        knots_angf = self.input.get("spline_knots", {}).get(
            "toroidal rotation", default_knots
        )
        fitter_angf = SplineFit(
            lower_bound=temp_angf.min() * 0.9,
            upper_bound=temp_angf.max() * 1.1,
            knots=knots_angf,
        )
        results_angf = fitter_angf(self.rho, self.t, temp_angf)
        self.toroidal_rotation = results_angf[0]

        try:
            temp_conc = deepcopy(self.diagnostics["cxrs"]["conc"])
            temp_conc.attrs["datatype"] = deepcopy(
                self.diagnostics["hrts"]["te"].attrs["datatype"]
            )  # TEMP for SplineFit checks
            knots_conc = self.input.get("spline_knots", {}).get(
                "toroidal rotation", default_knots
            )
            fitter_conc = SplineFit(
                lower_bound=temp_conc.min() * 0.9,
                upper_bound=temp_conc.max() * 1.1,
                knots=knots_conc,
            )
            results_conc = fitter_conc(self.rho, self.t, temp_conc)
            self.cxrs_concentration = results_conc[0]
        except KeyError:
            self.cxrs_concentration = None

    def calculate_power_loss(self):
        self.fzt = {
            elem: concat(
                [
                    self.FA[elem](
                        Ne=self.electron_density.interp(t=time),
                        Te=self.electron_temperature.interp(t=time),
                        tau=time,
                    ).expand_dims("t", -1)
                    for time in self.t.values
                ],
                dim="t",
            )
            .assign_coords({"t": self.t.values})
            .assign_attrs(transform=self.flux_surface)
            for elem in self.ion_species
        }

        self.power_loss = {
            elem: concat(
                [
                    self.PL[elem](
                        Ne=self.electron_density.interp(t=time),
                        Te=self.electron_temperature.interp(t=time),
                        F_z_t=self.fzt[elem].sel(t=time, method="nearest"),
                    ).expand_dims("t", -1)
                    for time in self.t.values
                ],
                dim="t",
            )
            .assign_coords({"t": self.t.values})
            .assign_attrs(transform=self.flux_surface)
            for elem in self.ion_species
        }

        self.sxr_power_loss = {
            elem: concat(
                [
                    self.SXRPL[elem](
                        Ne=self.electron_density.interp(t=time),
                        Te=self.electron_temperature.interp(t=time),
                        F_z_t=self.fzt[elem].sel(t=time, method="nearest"),
                    ).expand_dims("t", -1)
                    for time in self.t.values
                ],
                dim="t",
            )
            .assign_coords({"t": self.t.values})
            .assign_attrs(transform=self.flux_surface)
            for elem in self.ion_species
        }

        self.q: DataArray = (
            concat(
                [
                    MeanCharge()(FracAbundObj=self.fzt[elem], element=elem)
                    for elem in self.ion_species
                ],
                dim="element",
            )
            .assign_coords({"element": self.ion_species})
            .assign_attrs(transform=self.flux_surface)
        )

    def _calculate_vtor_asymmetry(self, element: str) -> DataArray:
        toroidal_rotation = (
            self.toroidal_rotation.expand_dims(
                {"element": [element]}  # type:ignore
            )
            .fillna(0.0)
            .copy()
        )
        ion_temperature = (
            self.ion_temperature.expand_dims(
                {"element": [element]}  # type:ignore
            )
            .fillna(1.0)
            .copy()
        )
        asymmetry = AsymmetryParameter()
        asymmetry_parameter = asymmetry(
            toroidal_rotations=toroidal_rotation,
            ion_temperature=ion_temperature,
            main_ion=self.main_ion,
            impurity=element,
            Zeff=self.diagnostics["zeff"]["zefh"].interp(t=self.t),
            electron_temp=self.electron_temperature,
            # mean_charge=self.q.copy(),
        )
        return asymmetry_parameter.drop_vars(
            [
                val
                for val in asymmetry_parameter.coords.keys()
                if val not in asymmetry_parameter.dims
            ]
        )

    def _calculate_asymmetry_high_z(self) -> DataArray:
        return self._calculate_vtor_asymmetry(element=self.high_z)

    def _calculate_asymmetry_other_z(self) -> DataArray:
        return self._calculate_vtor_asymmetry(element=self.other_z)

    def initialise_default_values(self):
        self._n_high_z._data = xr.zeros_like(self.sxr_emissivity)
        self._n_zeff_el._data = xr.zeros_like(self.electron_density)
        self._n_zeff_el_extra._data = xr.zeros_like(self.electron_density)
        self._n_other_z._data = xr.zeros_like(self.sxr_emissivity)
        self._n_main_ion._data = xr.zeros_like(self.electron_density)

    def _calculate_n_high_z(self) -> DataArray:
        other_densities = xr.concat(
            [
                self.n_zeff_el,
                self.n_zeff_el_extra,
                self.n_other_z,
                self.n_main_ion,
            ],
            dim="element",
        ).assign_coords(
            {"element": [self.zeff_el, self.zeff_el_extra, self.other_z, self.main_ion]}
        )
        n_high_z_initial = self._calculate_initial_n_high_z(
            other_densities=other_densities
        )
        # TODO add check for VUV
        try:
            n_high_z_extrapolated, extrapolator = self._extrapolate_n_high_z(
                n_high_z=n_high_z_initial
            )
        except Exception as e:
            print("Error extrapolating n_high_z: {}".format(e))
            return n_high_z_initial
        try:
            return self._optimise_n_high_z(
                n_high_z=n_high_z_extrapolated, extrapolator=extrapolator
            )
        except Exception as e:
            print("Error optimizing perturbation: {}".format(e))
            return n_high_z_extrapolated

    def _calculate_initial_n_high_z(self, other_densities: DataArray) -> DataArray:
        other_power_loss = self.sxr_power_loss_charge_averaged.sel(
            {"element": other_densities.coords["element"]}, drop=True
        )
        n_high_z: DataArray = (
            (
                self.sxr_calibration_factor * self.sxr_emissivity  # type: ignore
                - self.electron_density
                * (other_densities * other_power_loss).sum("element")
            )
            / (
                self.electron_density
                * self.sxr_power_loss_charge_averaged.sel(
                    element=self.high_z, drop=True
                )
            )
        ).assign_attrs({"transform": self.sxr_emissivity.attrs["transform"]})
        return n_high_z

    def _extrapolate_n_high_z(self, n_high_z: DataArray):
        # if self.n_high_z is None:
        #     raise UserWarning(
        #         "n_high_z has not yet been calculated, nothing to extrapolate"
        #     )
        rho_deriv, theta_deriv = n_high_z.transform.convert_from_Rz(
            self.R, self.z, self.t
        )
        n_high_z_Rz = n_high_z.interp({"rho_poloidal": rho_deriv, "theta": theta_deriv})
        extrapolator = ExtrapolateImpurityDensity()
        (_, n_high_z_rho_theta, _,) = extrapolator(
            impurity_density_sxr=n_high_z_Rz.where(n_high_z_Rz > 0.0, other=1.0).fillna(
                1.0
            ),
            electron_density=self.electron_density,
            electron_temperature=self.electron_temperature,
            truncation_threshold=1.5e3,
            flux_surfaces=self.flux_surface,
            asymmetry_parameter=None,
            t=self.t,
        )
        # self.additional_data["calculate_n_high_z"] = {
        #     "extrapolator": extrapolator,
        #     "threshold_rho": extrapolator.threshold_rho,
        # }
        return n_high_z_rho_theta, extrapolator

    def _optimise_n_high_z(
        self, n_high_z: DataArray, extrapolator: ExtrapolateImpurityDensity
    ) -> DataArray:
        # if self.additional_data.get("calculate_n_high_z", None) is None:
        #     raise UserWarning(
        #         "extrapolate_n_high_z has not yet been run,"
        #         " required to optimise n_high_z"
        #     )
        self.calculate_derived_bolometry()
        R_deriv: DataArray = self.flux_surface.convert_to_Rz(
            self.rho, self.theta, self.t
        )[0]
        return (
            extrapolator.optimize_perturbation(
                extrapolated_smooth_data=n_high_z.interp(
                    {"rho_poloidal": self.rho, "theta": self.theta}
                ),
                orig_bolometry_data=self.diagnostics["bolo"]["kb5v"],
                bolometry_obj=deepcopy(self.bolo_derivation),
                impurity_element=self.high_z,
                asymmetry_parameter=self.sxr_fitted_asymmetry_parameter.interp(
                    {"rho_poloidal": self.rho}
                ),
                threshold_rho=extrapolator.threshold_rho,
                R_deriv=R_deriv,
            )
            .drop_vars("element", errors="ignore")
            .assign_attrs({"transform": self.flux_surface})
        )

    def _calculate_n_zeff_el(self) -> DataArray:
        zeff = self.diagnostics["zeff"]["zefh"].interp(t=self.t.values)
        densities = xr.concat(
            [
                self.n_high_z,
                xr.zeros_like(self.electron_density).expand_dims({"theta": self.theta}),
                self.n_zeff_el_extra.expand_dims({"theta": self.theta}),  # type:ignore
                self.n_other_z,
            ],
            dim="element",
        ).assign_coords(
            {"element": [self.high_z, self.zeff_el, self.zeff_el_extra, self.other_z]}
        )
        conc_zeff_el: DataArray = ImpurityConcentration()(
            element=self.zeff_el,
            Zeff_LoS=zeff,
            impurity_densities=densities.where(densities >= 0.0, other=0.0).fillna(0.0),
            electron_density=self.electron_density.where(
                self.electron_density > 0.0, other=1.0
            ),
            mean_charge=self.q.fillna(0.0),
            flux_surfaces=self.flux_surface,
        )[0]
        self.additional_data["calculate_n_zeff_el"] = {"concentration": conc_zeff_el}
        n_zeff_el = conc_zeff_el.values * self.electron_density
        return n_zeff_el.assign_attrs({"transform": self.flux_surface})  # type: ignore

    def _calculate_n_zeff_el_extra(self) -> DataArray:
        concentration = 0.0  # TODO fit fixed concentration value from Zeff
        return (concentration * self.electron_density).assign_attrs(  # type: ignore
            {"transform": self.flux_surface}
        )

    def _calculate_n_other_z(self) -> DataArray:
        return xr.zeros_like(self.sxr_emissivity)  # TODO

    def _calculate_n_main_ion(self) -> DataArray:
        densities = xr.concat(
            [
                self.n_high_z,
                self.n_zeff_el.expand_dims({"theta": self.theta}),  # type:ignore
                self.n_zeff_el_extra.expand_dims({"theta": self.theta}),  # type:ignore
                self.n_other_z,
            ],
            dim="element",
        ).assign_coords(
            {"element": [self.high_z, self.zeff_el, self.zeff_el_extra, self.other_z]}
        )
        return MainIonDensity()(
            impurity_densities=densities.where(densities >= 0.0, other=0.0).fillna(0.0),
            electron_density=self.electron_density,
            mean_charge=self.q.where(self.q.element != self.main_ion, drop=True),
        ).assign_attrs({"transform": self.flux_surface})

    def _calibrate_sxr_emissivity(self) -> float:
        threshold_rho = recover_threshold_rho(
            truncation_threshold=1.5e3, electron_temperature=self.electron_temperature
        )
        derived_emissivity = (
            self.ion_densities
            * self.electron_density
            * self.sxr_power_loss_charge_averaged
        ).sum("element")
        calibration_factor = derived_emissivity / self.sxr_emissivity
        calibration_factor = calibration_factor.where(
            calibration_factor.coords["rho_poloidal"] < threshold_rho, drop=True
        ).mean(skipna=True)
        if calibration_factor == np.inf:
            raise UserWarning(
                "SXR calibration factor calculated as infinity, check data"
            )
        return calibration_factor

    def _calculate_derived_bolometry(self) -> DataArray:
        if not hasattr(self, "bolo_derivation"):
            impurity_elements = [
                val for val in self.ion_species if val != self.main_ion
            ]

            self.bolo_derivation = BolometryDerivation(
                flux_surfs=self.flux_surface,
                LoS_bolometry_data=bolo_los(self.diagnostics["bolo"]["kb5v"]),
                t_arr=self.t,
                impurity_densities=self.ion_densities.transpose(
                    "element", "rho_poloidal", "theta", "t"
                ),
                frac_abunds=[self.fzt.get(element) for element in impurity_elements],
                impurity_elements=impurity_elements,
                electron_density=self.electron_density,
                main_ion_power_loss=self.power_loss_charge_averaged.sel(
                    element=self.main_ion
                ),
                impurities_power_loss=self.power_loss_charge_averaged.sel(
                    {"element": self.ion_densities.coords["element"]}, drop=True
                ),
            )
            self.bolo_derivation(trim=True)
        else:
            self.bolo_derivation.impurity_densities = self.ion_densities.transpose(
                "element", "rho_poloidal", "theta", "t"
            )
        return self.bolo_derivation(deriv_only=True, trim=False)
