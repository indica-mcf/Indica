import sys
# Add Indica to python path
path_to_indica = '/home/jonathan.wood/git_home/Indica/'
sys.path.append(path_to_indica)

from indica.readers.read_st40 import ReadST40
from indica.models.plasma import example_run
from indica.models.neutral_beam import NeutralBeam
from indica.models.charge_exchange import ChargeExchange


# Workflow for running FIDASIM
def run_fidasim(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    call_fidasim: bool = False,
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

    print('Yo you made it!')

    # ToDo: activate this method
    # Call FIDASIM
    #cxspec(method='fidasim', run_fidasim=call_fidasim)

    return


if __name__ == "__main__":
    # Example run
    pulse = 11097
    tstart = 0.09
    tend = 0.10
    dt = 0.01
    call_fidasim = True

    run_fidasim(pulse, tstart, tend, dt, call_fidasim=call_fidasim)
