import sys
# Add Indica to python path
path_to_indica = '/home/jonathan.wood/git_home/Indica/'
sys.path.append(path_to_indica)

from indica.readers.read_st40 import ReadST40
from indica.models.plasma import example_run
from indica.models.neutral_beam import NeutralBeam
from indica.models.charge_exchange import ChargeExchange

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Workflow for running FIDASIM
def run_fidasim(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    call_fidasim: bool = False,
    plot: bool = True,
):

    # Read ST40 data
    st40 = ReadST40(tstart=tstart, tend=tend, dt=dt, pulse=pulse)
    st40(["cxff_pi"])

    # Get plasma
    plasma = example_run(tstart=tstart, tend=tend, dt=dt, pulse=pulse)
    plasma.set_equilibrium(st40.equilibrium)

    # Load Neutral Beam
    beam = NeutralBeam()

    # Get los transform and transect
    los_transform = st40.raw_data["cxff_pi"]["ti"].los_transform
    transect_transform = st40.raw_data["cxff_pi"]["ti"].transform

    # Create CX Spectrometer class
    cxspec = ChargeExchange(name="pi")
    cxspec.set_plasma(plasma)
    cxspec.set_beam(beam)
    cxspec.set_los_transform(los_transform)
    cxspec.set_transect_transform(transect_transform)

    # Call FIDASIM
    cxspec(method='fidasim', run_fidasim=call_fidasim)

    # Plotting
    if plot:
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))

        plt.close('all')

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.toroidal_rotation.sel(t=t, element='c').plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Vtor = cxspec.Vtor_at_channels.sel(t=t, method="nearest")
            omega_tor = Vtor * 1e3 / cxspec.transect_transform.R.values
            plt.scatter(
                Vtor.rho_poloidal, omega_tor, color=cols_time[i], marker="o", alpha=0.7
            )
        plt.xlabel("rho")
        plt.ylabel("Measured toroidal rotation (rad/s)")
        plt.ylim([0.0, 0.6e6])
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.ion_temperature.sel(t=t, element='c').plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Ti = cxspec.Ti_at_channels.sel(t=t, method="nearest")
            plt.scatter(Ti.rho_poloidal, Ti, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("rho")
        plt.ylabel("Measured ion temperature (eV)")
        plt.ylim([0.0, 7000])
        plt.legend()

        plt.show(block=True)

    return


if __name__ == "__main__":
    # Example run
    pulse = 10009
    tstart = 0.05
    tend = 0.06
    dt = 0.01
    call_fidasim = True

    run_fidasim(pulse, tstart, tend, dt, call_fidasim=call_fidasim)
