import warnings

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from indica import Equilibrium
from indica.configs.operators import AuroraConfig
from indica.utilities import format_dataarray
from indica.utilities import set_plot_colors
from .abstract_fractionalabundance import FractionalAbundance

try:
    import aurora
    from omfit_classes import omfit_eqdsk
except ImportError:
    pass

np.set_printoptions(edgeitems=10, linewidth=100)

CM, COLS = set_plot_colors()


class FractionalAbundanceAurora(FractionalAbundance):
    def _set_geqdsk(
        self,
        t_point: float,
    ):
        assert isinstance(self.equilibrium, Equilibrium)
        geqdsk_filepath = self.equilibrium.write_to_geqdsk(t_point=t_point)
        self.geqdsk = omfit_eqdsk.OMFITgeqdsk(geqdsk_filepath)

    def _set_kinetic_profiles(
        self,
        Te: DataArray,
        Ne: DataArray,
        Nh: DataArray,
    ):
        assert self.aurora_config
        kp = self.aurora_config["kin_profs"]
        kp["Te"]["rhop"] = Te.rhop.values
        kp["ne"]["rhop"] = Ne.rhop.values
        kp["n0"]["rhop"] = Nh.rhop.values
        kp["Te"]["times"] = np.atleast_1d(Te.t.values)
        kp["ne"]["times"] = np.atleast_1d(Ne.t.values)
        kp["n0"]["times"] = np.atleast_1d(Nh.t.values)
        kp["Te"]["vals"] = Te.values
        kp["ne"]["vals"] = Ne.values * 1e-6  # m^-3 -> cm^-3
        kp["n0"]["vals"] = Nh.values * 1e-6

    def _set_transport_profiles(
        self,
        D_z: DataArray,
        V_z: DataArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Interp DataArrays to Aurora rhop while converting to cm^-2 and cm^-1 units
        assert self.asim
        _D_z = (
            D_z.interp(rhop=self.asim.rhop_grid, kwargs={"fill_value": "extrapolate"})
            * 1e4
        )
        _V_z = (
            V_z.interp(rhop=self.asim.rhop_grid, kwargs={"fill_value": "extrapolate"})
            * 1e2
        )
        if D_z.ndim == 2:
            _D_z = _D_z.transpose("rhop", "t").values
            _V_z = _V_z.transpose("rhop", "t").values
        elif D_z.ndim > 2:
            raise Exception("D_z and V_z must be 1D or 2D.")
        return _D_z, _V_z

    def _run_steady_state(
        self, D_z: np.ndarray, V_z: np.ndarray, plot: bool = False, **kwargs
    ):
        # return self.asim.run_aurora_steady(D_z, V_z, plot=plot, **kwargs)
        raise NotImplementedError

    def _run_time_evolution(
        self, D_z: np.ndarray, V_z: np.ndarray, plot: bool = False, **kwargs
    ):
        return self.asim.run_aurora(D_z, V_z, plot=plot, **kwargs)

    def prepare(self, 
                D_z: DataArray=None, 
                V_z: DataArray=None, 
                main_ion:str="d", 
                equilibrium:Equilibrium = None):
        assert getattr(self, "equilibrium") is not None

        self.D_z = D_z
        self.V_z = V_z
        for key in ["acd", "scd", "ccd"]:
            _adas_file = getattr(self, f"{key}_file")
            self.aurora_config[key] = _adas_file
        self.aurora_config["imp"] = self.element.lower().title()
        self.aurora_config["main_element"] = self.main_ion.lower().title()

        if np.all(self.Nn.values == 0) and self.aurora_config["cxr_flag"]:
            raise ValueError("Nh is zero but cxr_flag is True.")

        if np.any(self.Nn.values != 0):
            if not self.aurora_config["cxr_flag"]:
                warnings.warn(
                    "Nh is non-zero but cxr_flag is False,"
                    "charge exchange will not be included."
                )

        self._set_kinetic_profiles(
            self.Te,
            self.Ne,
            self.Nn,
        )

    def run(self):
        times = np.atleast_1d(
            self.Te.t.values
        )  # same behaviour for 0D and 1D time inputs
        nz_init = None
        Nq = []  # density of ion charge states
        _Nq = None

        for t_idx, time in enumerate(times):
            # Set time and equilibrium
            if t_idx == 0:  # burn in time to reach steady state before time evolution
                self.aurora_config["timing"]["times"] = np.array([-0.1, time])
            else:
                self.aurora_config["timing"]["times"] = np.array(
                    [times[t_idx - 1], time]
                )
            self._set_geqdsk(time)

            # Prepare
            self.asim = aurora.aurora_sim(
                namelist=self.aurora_config, geqdsk=self.geqdsk
            )
            _D_z, _V_z = self._set_transport_profiles(self.D_z, self.V_z)
            if t_idx != 0:
                # use previous results as initial conditions for next time step
                # Interpolate to match new rhop grid each time step
                nz_init = _Nq.interp(
                    rhop=self.asim.rhop_grid, kwargs={"fill_value": "extrapolate"}
                ).values

            # Run Aurora
            aurora_result = self._run_time_evolution(
                D_z=_D_z, V_z=_V_z, times_DV=self.D_z.t.values, nz_init=nz_init
            )

            # Refactor output
            _Nq = aurora_result["nz"][:, :, -1]
            _Nq[_Nq < 0] = 0  # Set negative values to zero (numerical issues)
            _Nq = xr.DataArray(
                data=_Nq,
                coords={
                    "rhop": self.asim.rhop_grid,
                    "ion_charge": np.arange(self.asim.Z_imp + 1),
                },
            )
            Nq.append(_Nq.interp(rhop=self.Te.rhop))
        Nq = xr.concat(Nq, pd.Index(times, name="t"))
        self.F_z_t = Nq / Nq.sum("ion_charge")
        self.ion_charge = np.arange(self.asim.Z_imp + 1)
        self.spatial_coord = self.Te.rhop
        self.t = times
        return self.F_z_t

    def refactor_output(self):
        coords = {}
        for dim, coord in self.F_z_t.coords.items():
            coords[dim] = coord.data
        self.F_z_t = format_dataarray(self.F_z_t, "fractional_abundance", coords)
        return self.F_z_t

    def plot_fractional_abundance(self):
        aurora.plot_tools.slider_plot(
            self.F_z_t.rhop,
            self.F_z_t.t,
            self.F_z_t.values.transpose(2, 1, 0),
            xlabel=r"$\rho$ [-]",
            ylabel="time [s]",
            zlabel=r"fractional abundance$ [-]",
            labels=map(str, range(self.F_z_t.ion_charge.values.shape[0])),
            plot_sum=True,
        )

    def __init__(
        self,
        element,
        scd_year=None,
        acd_year=None,
        ccd_year=None,
        main_ion:str="d",
        aurora_config: dict = AuroraConfig,
        equilibrium: Equilibrium = None,
    ):
        super().__init__(
            element,
            scd_year,
            acd_year,
            ccd_year,
            main_ion=main_ion,
            aurora_config=aurora_config,
            equilibrium=equilibrium,
        )

    def __call__(
        self,
        Te,
        Ne,
        Nn=None,
        D_z: DataArray = None,
        V_z: DataArray = None,
    ):
        return super().__call__(Te, Ne, Nn, D_z=D_z, V_z=V_z)
