from typing import cast
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from xarray import concat
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.converters.lines_of_sight import LinesOfSightTransform
from indica.operators.main_ion_density import MainIonDensity
from indica.operators.mean_charge import MeanCharge
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..utilities import input_check


class BolometryDerivation(Operator):
    """Class to hold relevant variables and functions relating to the derivation of
    bolometry data from plasma quantities (densities, temperatures, etc.)

    Parameters
    ----------
    flux_surfaces
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    LoS_bolometry_data
        Line-of-sight bolometry data in the same format as given in:
        tests/unit/operator/KB5_Bolometry_data.py
    t_arr
        Array of time values to interpolate the (rho, theta) grids on.
        xarray.DataArray with dimensions (t).
    impurity_densities
        Densities for all impurities
        (including the extrapolated smooth density of the impurity in question),
        xarray.DataArray with dimensions (elements, rho, theta, t).
    frac_abunds
        Fractional abundances list of fractional abundances
        (an xarray.DataArray for each impurity) dimensions of each element in
        the list are (ion_charges, rho, t).
    impurity_elements
        List of element symbols(as strings) for all impurities.
    electron_density
        xarray.DataArray of electron density, xarray.DataArray wit dimensions (rho, t)
    main_ion_power_loss
        Power loss associated with the main ion (eg. deuterium),
        xarray.DataArray with dimensions (rho, t)
    main_ion_density
        Density profile for the main ion,
        xarray.DataArray with dimensions (rho, theta, t)

    Methods
    -------
    __bolometry_coord_transforms()
        Transform the bolometry coords from LoS to (rho, theta) and (R, z).
    __bolometry_setup()
        Calculating main ion density for the bolometry derivation.
    __bolometry_channel_filter()
        Filters the bolometry data to reduce the number of channels by eliminating
        channels that are too close together.
    __bolometry_derivation(trim, t_val)
        Derive bolometry including the extrapolated smoothed impurity density.
    __call__(deriv_only, trim, t_val)
        Varying workflow to derive bolometry from plasma quantities.
        (Varying as in, if full setup and derivation is needed or only derivaiton.)

    Returns
    -------
    derived_power_loss_LoS_tot
        Total derived bolometric power loss values along all lines-of-sight.
        xarray.DataArray with dimensions (channels, t) or (channels) depending
        on whether t_val is provided.

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("plasma", "flux_surface_coordinates"),
        ("bolometric", "lines_of_sight_data"),
        ("plasma", "times"),
        ("impurities", "number_density"),
        ("impurities", "fractional_abundance"),
        ("impurities", "elements"),
        ("electrons", "number_density"),
        ("main_ion", "total radiated power loss"),
        ("main_ion", "number_density"),
    ]

    RESULT_TYPES: List[Union[DataType, EllipsisType]] = []

    def __init__(
        self,
        flux_surfs: FluxSurfaceCoordinates,
        LoS_bolometry_data: Sequence,
        t_arr: DataArray,
        impurity_densities: DataArray,
        frac_abunds: Sequence,
        impurity_elements: Sequence,
        electron_density: DataArray,
        main_ion_power_loss: DataArray,
        impurities_power_loss: DataArray,
        sess: session.Session = session.global_session,
    ):
        """Initialises the BolometryDerivation class. Checks all inputs for errors."""
        super().__init__(sess=sess)

        input_check(
            "flux_surfs",
            flux_surfs,
            FluxSurfaceCoordinates,
        )

        input_check("LoS_bolometry_data", LoS_bolometry_data, Sequence)

        input_check(
            "t_arr", t_arr, DataArray, ndim_to_check=1, greater_than_or_equal_zero=True
        )

        input_check(
            "impurity_densities",
            impurity_densities,
            DataArray,
            ndim_to_check=4,
            greater_than_or_equal_zero=True,
        )

        input_check("frac_abunds", frac_abunds, Sequence)

        input_check("impurity_elements", impurity_elements, Sequence)

        input_check(
            "electron_density",
            electron_density,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "main_ion_power_loss",
            main_ion_power_loss,
            DataArray,
            ndim_to_check=2,
            greater_than_or_equal_zero=True,
        )

        input_check(
            "impurities_power_loss",
            impurities_power_loss,
            DataArray,
            ndim_to_check=3,
            greater_than_or_equal_zero=True,
        )

        self.flux_surfaces = flux_surfs
        self.LoS_bolometry_data = LoS_bolometry_data
        self.t_arr = t_arr
        self.impurity_densities = impurity_densities
        self.frac_abunds = frac_abunds
        self.impurity_elements = impurity_elements
        self.electron_density = electron_density
        self.main_ion_power_loss = main_ion_power_loss
        self.impurities_power_loss = impurities_power_loss

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return super().return_types(*args)

    def __bolometry_coord_transforms(self):
        """Transform the bolometry coords from LoS to (rho, theta) and (R, z).

        Returns
        -------
        LoS_coords
            List of dictionaries containing the rho, theta, x and z arrays
            and dl for the resolution of the LoS coordinates.
        """
        LoS_coords = []
        for iLoS in range(len(self.LoS_bolometry_data)):
            LoS_transform = LinesOfSightTransform(*self.LoS_bolometry_data[iLoS])

            x1_name = LoS_transform.x1_name
            x2_name = LoS_transform.x2_name

            x1 = DataArray(
                data=np.array([0]), coords={x1_name: np.array([0])}, dims=[x1_name]
            )
            x2 = DataArray(
                data=np.linspace(0, 1, 30),
                coords={x2_name: np.linspace(0, 1, 30)},
                dims=[x2_name],
            )

            R_arr, z_arr = LoS_transform.convert_to_Rz(x1, x2, self.t_arr)

            rho_arr, theta_arr = self.flux_surfaces.convert_from_Rz(R_arr, z_arr)
            rho_arr = cast(DataArray, rho_arr).interp(t=self.t_arr, method="linear")
            theta_arr = cast(DataArray, theta_arr).interp(t=self.t_arr, method="linear")

            rho_arr = np.abs(rho_arr)
            rho_arr = rho_arr.assign_coords({x2_name: x2})
            rho_arr = rho_arr.drop(x1_name).squeeze()
            rho_arr = rho_arr.fillna(2.0)
            theta_arr = theta_arr.drop(x1_name).squeeze()

            dl = LoS_transform.distance(x2_name, DataArray(0), x2[0:2], 0)

            LoS_coords.append(
                dict(
                    {
                        # dimensions for rho_arr and
                        # theta_arr are (channel no., distance along LoS)
                        "rho": rho_arr,
                        "theta": theta_arr,
                        # dimensions for dl are (channel no.)
                        "dl": dl,
                        # dimensions for R_arr and
                        # z_arr are (channel no., distance along LoS)
                        "R": R_arr,
                        "z": z_arr,
                        # dimensions for t are (t)
                        "t": self.t_arr,
                    }
                )
            )

        self.LoS_coords = LoS_coords

        return LoS_coords

    def __bolometry_setup(self):
        """Calculating main ion density for the bolometry derivation.

        Returns
        -------
        main_ion_density
            Density profile for the main ion,
            xarray.DataArray with dimensions (rho, theta, t)
        """
        mean_charges = zeros_like(self.electron_density)
        mean_charges = mean_charges.data
        mean_charges = np.tile(mean_charges, (len(self.impurity_elements), 1, 1))
        # Ignoring mypy error since mypy refuses to acknowlege electron_density.coords
        # as a dictionary
        mean_charges_coords = {
            "elements": self.impurity_elements,  # type: ignore
            **self.electron_density.coords,  # type: ignore
        }
        mean_charges = DataArray(
            data=mean_charges,
            coords=mean_charges_coords,  # type:ignore
            dims=["elements", *self.electron_density.dims],
        )

        for ielement, element in enumerate(self.impurity_elements):
            mean_charge = MeanCharge()
            mean_charge = mean_charge(self.frac_abunds[ielement], element)
            mean_charges.loc[element] = mean_charge

        main_ion_density_obj = MainIonDensity()
        main_ion_density = main_ion_density_obj(
            self.impurity_densities, self.electron_density, mean_charges
        )

        main_ion_density = main_ion_density.transpose("rho", "theta", "t")

        self.main_ion_density = main_ion_density

        return main_ion_density

    def __bolometry_channel_filter(self):
        """Filters the bolometry data to reduce the number of channels by eliminating
        channels that are too close together.

        Returns
        -------
        LoS_bolometry_data_trimmed
            Lines-of-sight data with a reduced number of channels. List with
            the same formatting as self.LoS_bolometry_data.
        LoS_coords_trimmed
            LoS coordinates (as returned by bolometry_coord_transforms) with a
            reduced number of channels. List with the same formatting as
            self.LoS_coords.
        """
        LoS_bolometry_data_in = self.LoS_bolometry_data
        LoS_coords_in = self.LoS_coords

        LoS_bolometry_data = []
        LoS_coords = []

        t_arr = LoS_coords_in[0]["t"]

        R_central = self.flux_surfaces.equilibrium.rmag.interp(t=t_arr, method="linear")

        LoS_R_midplane_multi = []

        for iLoS in range(len(LoS_bolometry_data_in)):
            R_arr = LoS_coords_in[iLoS]["R"]
            z_arr = LoS_coords_in[iLoS]["z"]

            try:
                midplane_LoS_pos = z_arr.indica.invert_interp(
                    values=0.0, target=z_arr.dims[1], method="nearest"
                )
            except ValueError:
                midplane_LoS_pos = 2.0

            LoS_R_midplane = R_arr.interp(
                {R_arr.dims[1]: midplane_LoS_pos}, method="nearest"
            )

            if (LoS_R_midplane > R_central).all():
                LoS_bolometry_data.append(LoS_bolometry_data_in[iLoS])
                LoS_coords.append(LoS_coords_in[iLoS])
                LoS_R_midplane_multi.append(LoS_R_midplane.data[0])

        LoS_R_midplane_positions = DataArray(
            data=LoS_R_midplane_multi,
            coords={"channels": [j[6] for j in LoS_bolometry_data]},
            dims=["channels"],
        )

        LoS_R_midplane_positions_sorted = LoS_R_midplane_positions.sortby(
            LoS_R_midplane_positions
        )

        radial_resolution_threshold = 0.1  # in metres

        LoS_R_midplane_positions_diff = LoS_R_midplane_positions_sorted.diff(
            "channels", label="upper"
        )

        LoS_R_midplane_positions_diff_first = DataArray(
            data=np.array([True]),
            coords={
                "channels": np.array(
                    [LoS_R_midplane_positions_sorted.coords["channels"].data[0]]
                )
            },
            dims=["channels"],
        )

        LoS_R_midplane_positions_diff = concat(
            [LoS_R_midplane_positions_diff_first, LoS_R_midplane_positions_diff],
            dim="channels",
        )

        LoS_R_midplane_trim_mask = (
            LoS_R_midplane_positions_diff > radial_resolution_threshold
        )

        LoS_R_midplane_trimmed = LoS_R_midplane_positions_sorted[
            LoS_R_midplane_trim_mask
        ]

        LoS_bolometry_data_trimmed = []
        LoS_coords_trimmed = []
        for iLoS in range(len(LoS_bolometry_data)):
            orig_channel = LoS_bolometry_data[iLoS][6]
            if orig_channel in LoS_R_midplane_trimmed.coords["channels"].data:
                LoS_bolometry_data_trimmed.append(LoS_bolometry_data[iLoS])
                LoS_coords_trimmed.append(LoS_coords[iLoS])

        self.LoS_bolometry_data_trimmed, self.LoS_coords_trimmed = (
            LoS_bolometry_data_trimmed,
            LoS_coords_trimmed,
        )

        return LoS_bolometry_data_trimmed, LoS_coords_trimmed

    def __bolometry_derivation(
        self,
        trim: bool = False,
        t_val: float = None,
    ):
        """Derive bolometry including the extrapolated smoothed impurity density.

        Parameters
        ----------
        trim
            Boolean specifying whether the number of bolometry channels are trimmed.
        t_val
            Optional time value(float) for which to calculate the bolometry data.

        Returns
        -------
        derived_power_loss_LoS_tot
            Total derived bolometric power loss values along all lines-of-sight.
            xarray.DataArray with dimensions (channels, t) or (channels) depending
            on whether t_val is provided.
        """
        if trim:
            if not (
                hasattr(self, "LoS_bolometry_data_trimmed")
                and hasattr(self, "LoS_coords_trimmed")
            ):
                return AttributeError(
                    'Argument "trim" is set to True but bolometry_channel_filter() \
                        has not yet been run at least once.'
                )
            LoS_bolometry_data = self.LoS_bolometry_data_trimmed
            LoS_coords_in = self.LoS_coords_trimmed
        else:
            LoS_bolometry_data = self.LoS_bolometry_data
            LoS_coords_in = self.LoS_coords

        if t_val is not None:
            LoS_coords = cast(Sequence, [{} for i in LoS_coords_in])
            impurity_densities = self.impurity_densities.sel(t=t_val)
            main_ion_power_loss = self.main_ion_power_loss.sel(t=t_val)
            impurities_power_loss = self.impurities_power_loss.sel(t=t_val)
            electron_density = self.electron_density.sel(t=t_val)
            main_ion_density = self.main_ion_density.sel(t=t_val)
            for icoord in range(len(LoS_coords_in)):
                LoS_coords[icoord]["rho"] = LoS_coords_in[icoord]["rho"].sel(t=t_val)
                LoS_coords[icoord]["theta"] = LoS_coords_in[icoord]["theta"].sel(
                    t=t_val
                )
                LoS_coords[icoord]["dl"] = LoS_coords_in[icoord]["dl"]
                LoS_coords[icoord]["R"] = LoS_coords_in[icoord]["R"]
                LoS_coords[icoord]["z"] = LoS_coords_in[icoord]["z"]
        else:
            LoS_coords = LoS_coords_in
            impurity_densities = self.impurity_densities
            main_ion_power_loss = self.main_ion_power_loss
            impurities_power_loss = self.impurities_power_loss
            electron_density = self.electron_density
            main_ion_density = self.main_ion_density

        derived_power_loss = electron_density * (main_ion_density * main_ion_power_loss)
        impurities_losses = impurity_densities * impurities_power_loss
        impurities_losses = impurities_losses.sum(dim="elements")
        impurities_losses *= electron_density
        derived_power_loss += impurities_losses

        if t_val is not None:
            derived_power_loss = derived_power_loss.transpose("rho", "theta")

            derived_power_loss_LoS_tot = DataArray(
                data=np.zeros((len(LoS_bolometry_data))),
                coords={
                    "channels": np.linspace(
                        0,
                        len(LoS_bolometry_data),
                        len(LoS_bolometry_data),
                        endpoint=False,
                    ),
                },
                dims=["channels"],
            )

        else:
            derived_power_loss = derived_power_loss.transpose("t", "rho", "theta")

            t_arr = derived_power_loss.coords["t"]

            derived_power_loss_LoS_tot = DataArray(
                data=np.zeros((len(LoS_bolometry_data), t_arr.shape[0])),
                coords={
                    "channels": np.linspace(
                        0,
                        len(LoS_bolometry_data),
                        len(LoS_bolometry_data),
                        endpoint=False,
                    ),
                    "t": t_arr,
                },
                dims=["channels", "t"],
            )

        for iLoS in range(len(LoS_bolometry_data)):
            LoS_transform = LinesOfSightTransform(*LoS_bolometry_data[iLoS])

            x2_name = LoS_transform.x2_name

            rho_arr = LoS_coords[iLoS]["rho"]
            theta_arr = LoS_coords[iLoS]["theta"]

            derived_power_loss_LoS = derived_power_loss.interp(
                {"rho": rho_arr, "theta": theta_arr}
            )

            derived_power_loss_LoS = derived_power_loss_LoS.fillna(0.0)

            dl = LoS_coords[iLoS]["dl"]
            dl = cast(DataArray, dl)[1]

            derived_power_loss_LoS = derived_power_loss_LoS.sum(dim=x2_name) * dl
            derived_power_loss_LoS_tot[iLoS] = derived_power_loss_LoS.squeeze()

        self.derived_power_loss_LoS_tot = derived_power_loss_LoS_tot

        return derived_power_loss_LoS_tot

    def __call__(  # type: ignore
        self, deriv_only: bool = False, trim: bool = True, t_val: float = None
    ):
        """Varying workflow to derive bolometry from plasma quantities.
        (Varying as in, if full setup and derivation is needed or only derivaiton.)

        Parameters
        ----------
        deriv_only
            Optional boolean specifying if only derivation is needed(True) or if full
            setup and derivation is needed(False).
        trim
            Optional boolean specifying whether to use bolometry data with trimmed
            channels(True) or not(False).
        t_val
            Optional time value for which to calculate the bolometry data.
            (This is passed to bolometry_derivation())

        Returns
        -------
        derived_power_loss_LoS_tot
            Total derived bolometric power loss values along all lines-of-sight.
            xarray.DataArray with dimensions (channels, t) or (channels) depending
            on whether t_val is provided.
        """
        input_check("deriv_only", deriv_only, bool)
        input_check("trim", trim, bool)

        if t_val is not None:
            input_check("t_val", t_val, float, greater_than_or_equal_zero=True)

        if not deriv_only:
            self.__bolometry_coord_transforms()

            self.__bolometry_setup()

            if trim:
                self.__bolometry_channel_filter()

        self.__bolometry_derivation(trim, t_val)

        return self.derived_power_loss_LoS_tot
