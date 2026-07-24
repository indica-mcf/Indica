import copy

import matplotlib.pylab as plt
import numpy as np
import scipy
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from indica.utilities import format_coord
from indica.utilities import format_dataarray
from indica.utilities import set_plot_colors
from .abstract_fractionalabundance import FractionalAbundance

CM, COLS = set_plot_colors()


class FractionalAbundanceAdas(FractionalAbundance):
    def _interpolate_rates(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Interpolates rates based on inputted Ne and Te, also determines the number
        of ionisation charges for a given element.

        Ne
            electron density profile
        Te
            electron temperature profile
        """

        self.Ne, self.Te = Ne, Te  # type: ignore

        scd_spec = self.scd.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        acd_spec = self.acd.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        if self.ccd is not None:
            ccd_spec = self.ccd.indica.interp2d(
                electron_temperature=Te,
                electron_density=Ne,
                method="cubic",
                assume_sorted=True,
            )
        else:
            ccd_spec = xr.full_like(scd_spec, 0.0)

        self.scd_spec, self.acd_spec, self.ccd_spec = scd_spec, acd_spec, ccd_spec
        self.nq = len(self.scd_spec.ion_charge) + 1
        self.ion_charge = np.linspace(0, self.nq - 1, self.nq)

        return scd_spec, acd_spec, ccd_spec, self.nq

    def _calc_ionisation_balance_matrix(
        self,
        Ne: DataArray,
        Nh: DataArray = None,
    ):
        """Calculates the ionisation balance matrix
        Ne
            electron density profile
        Nh
            thermal neutral hydrogen profile
        """
        if Nh is not None:
            if self.ccd is None:
                raise ValueError("Nh cannot be given if ccd is None.")
        else:
            Nh = xr.full_like(Ne, 0.0)

        self.Ne, self.Nh = Ne, Nh  # type: ignore

        scd, acd, ccd = self.scd_spec, self.acd_spec, self.ccd_spec

        # Additional coordinate that is not ion_charge (e.g. rhop)
        coord = scd.coords[[k for k in scd.dims if k != "ion_charge"][0]]
        self.coord = coord
        self.dim = coord.dims[0]
        self.ncoord = len(coord)

        ionisation_balance_matrix = np.zeros((self.nq, self.nq, self.ncoord))

        q = 0
        ionisation_balance_matrix[q, q : q + 2] = np.array(
            [
                -Ne * scd.sel(ion_charge=q),
                Ne * acd.sel(ion_charge=q) + Nh * ccd.sel(ion_charge=q),
            ]
        )

        for q in range(1, self.nq - 1):
            ionisation_balance_matrix[q, q - 1 : q + 2] = np.array(
                [
                    Ne * scd.sel(ion_charge=q - 1),
                    -Ne * (scd.sel(ion_charge=q) + acd.sel(ion_charge=q - 1))
                    - Nh * ccd.sel(ion_charge=q - 1),
                    Ne * acd.sel(ion_charge=q) + Nh * ccd.sel(ion_charge=q),
                ]
            )

        q = self.nq - 1
        ionisation_balance_matrix[q, q - 1 : q + 1] = np.array(
            [
                Ne * scd.sel(ion_charge=q - 1),
                -Ne * acd.sel(ion_charge=q - 1) - Nh * ccd.sel(ion_charge=q - 1),
            ]
        )

        ionisation_balance_matrix = np.squeeze(ionisation_balance_matrix)
        self.ionisation_balance_matrix = ionisation_balance_matrix

        return ionisation_balance_matrix

    def _calc_F_z_tinf(
        self,
    ):
        """Calculates the equilibrium fractional abundance of all ionisation charges,
        F_z(t=infinity) used for the final time evolution equation.
        """
        ionisation_balance_matrix = self.ionisation_balance_matrix

        null_space = np.zeros((self.nq, self.ncoord))
        F_z_tinf = np.zeros((self.nq, self.ncoord))

        for ix1 in range(self.ncoord):
            null_space[:, ix1, np.newaxis] = scipy.linalg.null_space(
                ionisation_balance_matrix[:, :, ix1]
            )

        # Complex type casting for compatibility with eigen calculation results later.
        F_z_tinf = np.abs(null_space).astype(dtype=np.complex128)

        # normalization needed for high-z elements
        F_z_tinf = F_z_tinf / np.sum(F_z_tinf, axis=0)

        self.F_z_tinf = F_z_tinf

        return np.real(F_z_tinf)

    def _calc_eigen_vals_and_vecs(
        self,
    ):
        """Calculates the eigenvalues and eigenvectors of the ionisation balance
        matrix.
        """
        eig_vals = np.zeros((self.nq, self.ncoord), dtype=np.complex128)
        eig_vecs = np.zeros(
            (self.nq, self.nq, self.ncoord),
            dtype=np.complex128,
        )

        for ix1 in range(self.ncoord):
            eig_vals[:, ix1], eig_vecs[:, :, ix1] = scipy.linalg.eig(
                self.ionisation_balance_matrix[:, :, ix1],
            )

        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs

        return eig_vals, eig_vecs

    def _calc_eigen_coeffs(
        self,
        F_z_t0: DataArray = None,
    ):
        """Calculates the coefficients from the eigenvalues and eigenvectors for the
        time evolution equation.

        F_z_t0
            Initial fractional abundance for given impurity element. (Optional)
        """

        if F_z_t0 is None:
            # mypy doesn't understand contionals or reassignments either.
            F_z_t0 = np.zeros(self.F_z_tinf.shape, dtype=np.complex128)  # type: ignore
            F_z_t0[0, :] = np.array(  # type: ignore
                [1.0 + 0.0j for i in range(self.ncoord)]
            )

            F_z_t0 = DataArray(
                data=F_z_t0,
                coords={"ion_charge": self.ion_charge, self.dim: self.coord},
            )
        else:
            try:
                assert F_z_t0.ndim < 3
            except AssertionError:
                raise ValueError("F_z_t0 must be at most 2-dimensional.")

            F_z_t0 = F_z_t0 / np.sum(F_z_t0, axis=0)
            F_z_t0 = F_z_t0.as_type(dtype=np.complex128)  # type: ignore

        eig_vals = self.eig_vals
        eig_vecs_inv = np.zeros(self.eig_vecs.shape, dtype=np.complex128)
        for ix1 in range(self.ncoord):
            eig_vecs_inv[:, :, ix1] = np.linalg.pinv(
                np.transpose(self.eig_vecs[:, :, ix1])
            )

        boundary_conds = F_z_t0 - self.F_z_tinf

        eig_coeffs = np.zeros(eig_vals.shape, dtype=np.complex128)
        for ix1 in range(self.ncoord):
            eig_coeffs[:, ix1] = np.dot(boundary_conds[:, ix1], eig_vecs_inv[:, :, ix1])

        self.eig_coeffs = eig_coeffs
        self.F_z_t0 = np.abs(np.real(F_z_t0))
        return self.eig_coeffs, self.F_z_t0

    def _calculate_abundance(self, tau: LabeledArray):
        """Calculates the fractional abundance of all ionisation charges at time tau.

        tau
            Time after t0 (t0 is defined as the time at which F_z_t0 is taken).
        """
        F_z_t = copy.deepcopy(self.F_z_tinf)
        for ix1 in range(self.ncoord):
            if isinstance(tau, (DataArray, np.ndarray)):
                itau = tau[ix1].values if isinstance(tau, DataArray) else tau[ix1]
            else:
                itau = tau

            for q in range(self.nq):
                F_z_t[:, ix1] += (
                    self.eig_coeffs[q, ix1]
                    * np.exp(self.eig_vals[q, ix1] * itau)
                    * self.eig_vecs[:, q, ix1]
                )

        self.F_z_t = np.abs(np.real(F_z_t))
        self.tau = tau

        return F_z_t

    def prepare(self, tau: LabeledArray = None, F_z_t0: DataArray = None):
        self.tau = tau
        self.F_z_t0 = F_z_t0
        self._interpolate_rates(self.Ne, self.Te)

    def run(self):
        # TODO: implement loop for multiple time-points similar to Aurora
        self._calc_ionisation_balance_matrix(self.Ne, self.Nn)
        self._calc_F_z_tinf()

        if self.tau is not None:
            self._calc_eigen_vals_and_vecs()
            self._calc_eigen_coeffs(self.F_z_t0)
            F_z_t = self._calculate_abundance(self.tau)
            self.F_z_t = F_z_t
        else:
            F_z_t = np.real(self.F_z_tinf)

        self.F_z_t = F_z_t

        return F_z_t

    def refactor_output(self):
        _ion_charge = format_coord(self.ion_charge, "ion_charge")
        spatial_coord = self.coord
        for key in spatial_coord.coords.keys():
            if key not in spatial_coord.dims:
                spatial_coord = spatial_coord.drop_vars(key)
        coords = {"ion_charge": _ion_charge, self.dim: spatial_coord}
        if self.F_z_t0 is not None:
            self.F_z_t0 = format_dataarray(self.F_z_t0, "fractional_abundance", coords)
        self.F_z_t = format_dataarray(self.F_z_t, "fractional_abundance", coords)

        return self.F_z_t

    def plot_fractional_abundance(self):
        cols = CM(np.linspace(0.1, 0.75, len(self.ion_charge), dtype=float))

        for iq in np.int_(self.ion_charge):
            self.F_z_t.sel(ion_charge=iq).plot(color=cols[iq], alpha=0.8, label=iq)
        plt.legend()
        plt.xlim(0, 1.2)
        plt.title(f"{self.element.title()} fractional abundance")

    def __call__(self, Te, Ne, Nn=None, tau: DataArray = None):
        return super().__call__(Te, Ne, Nn, tau=tau)
