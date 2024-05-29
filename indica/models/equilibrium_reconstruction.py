import matplotlib.pyplot as plt
import xarray as xr

from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.defaults.load_defaults import load_default_objects
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import assign_datatype
from indica.utilities import check_time_present


class EquilibriumReconstruction(AbstractDiagnostic):
    """
    Object representing observations from a magnetic reconstruction
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_equilibrium",
    ):
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "wp":
                self.bckc[quant] = self.wp
                error = xr.full_like(self.bckc[quant], 0.0)
                stdev = xr.full_like(self.bckc[quant], 0.0)
                self.bckc[quant].attrs = {
                    "error": error,
                    "stdev": stdev,
                }
                assign_datatype(self.bckc[quant], datatype)
            else:
                # print(f"{quant} not available in model for {self.instrument_method}")
                continue

    def __call__(
        self,
        t=None,
        **kwargs,
    ):
        """

        Returns
        -------
        bckc values
        """
        if self.plasma is None:
            raise ValueError("plasma object is needed")

        if t is None:
            t = self.plasma.time_to_calculate

        check_time_present(t, self.plasma.wp.t)

        self.wp = self.plasma.wp.interp(t=t)
        self._build_bckc_dictionary()
        return self.bckc


def example_run(
    diagnostic_name: str = "efit",
    plasma=None,
    plot=False,
    t=None,
):
    if plasma is None:
        from indica.equilibrium import fake_equilibrium

        plasma = load_default_objects("st40", "plasma")
        machine_dims = plasma.machine_dimensions
        equilibrium = fake_equilibrium(
            tstart=plasma.tstart,
            tend=plasma.tend,
            dt=plasma.dt / 2.0,
            machine_dims=machine_dims,
        )
        plasma.set_equilibrium(equilibrium)

    model = EquilibriumReconstruction(diagnostic_name)
    model.set_plasma(plasma)
    bckc = model()

    if plot:
        bckc["wp"].plot()

    return plasma, model, bckc


if __name__ == "__main__":
    example_run(plot=True)
    plt.show(block=True)
