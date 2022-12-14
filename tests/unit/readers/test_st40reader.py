import matplotlib.pylab as plt
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.readers import ST40Reader

PULSE = 9229
TSTART = 0.01
TEND = 0.1

READER = ST40Reader(PULSE, TSTART, TEND)
EQUILIBRIUM_DATA = READER.get("", "efit", 0)
EQUILIBRIUM = Equilibrium(EQUILIBRIUM_DATA)
FLUX_TRANSFORM = FluxSurfaceCoordinates("poloidal")
FLUX_TRANSFORM.set_equilibrium(EQUILIBRIUM)

INSTRUMENT_INFO = {
    "xrcs": ("sxr", "xrcs", 0, set()),
    "brems": ("spectrom", "lines", -1, ["brems"]),
    "smmh1": ("interferom", "smmh1", 0, set()),
    "nirh1": ("interferom", "nirh1", 0, set()),
    "efit": ("", "efit", 0, set()),
}

# "sxr_camera": ("sxr", "diode_arrays", 0, ["filter_4"]),


def run_reader_get_methods(
    instrument_name: str,
    mds_only=False,
    plot=False,
):
    """
    General test script to read data from MDS+ and calculate LOS information
    including Cartesian-flux surface mapping

    TODO: currently only runs, but no assertions to check what data it returns

    Parameters
    ----------
    instrument_name
        Key from INSTRUMENT_INFO to give all the necessary inputs
    mds_only
        Returns only ST40Reader database dictionary. Otherwise returns also
        data structure crunched by the abstractreader
    plot
        Plot lines of sight and mapping on equilibrium reconstruction

    Returns
    -------

    """
    if instrument_name not in INSTRUMENT_INFO:
        raise ValueError(
            f"Instrument not available. Possible choices: {list(INSTRUMENT_INFO)}"
        )

    uid, instrument, revision, quantities = INSTRUMENT_INFO[instrument_name]

    print(f"Reading {uid}, {instrument}, {revision}")
    instrument_method = READER.INSTRUMENT_METHODS[instrument]
    database_quantities = READER.available_quantities(instrument)
    if quantities:
        database_quantities = quantities

    database_results = getattr(READER, f"_{instrument_method}")(
        uid,
        instrument,
        revision,
        database_quantities,
    )

    if mds_only:
        return database_results

    data = READER.get(uid, instrument, revision, set(quantities))

    quantities = list(data)
    trans = data[quantities[0]].transform
    if hasattr(trans, "set_flux_transform"):
        trans.set_flux_transform(FLUX_TRANSFORM)
        trans._convert_to_rho(t=np.array([0.02, 0.03, 0.04]))
        if plot:
            trans.plot_los()

    return data, database_results


def test_all(interactive=False, plot=False):
    plt.ion()
    for instrument_name in INSTRUMENT_INFO.keys():
        print(f"\n Testing {instrument_name} \n")
        data, database_resutls = run_reader_get_methods(instrument_name, plot=plot)

        for quant in data.keys():
            if hasattr(data[quant], "transform"):
                if "LineOfSightTransform" in str(data[quant].transform):
                    if "line_of_sight_multi" not in str(data[quant].transform):
                        raise ValueError(
                            f"{instrument_name}:{quant} using"
                            f" \n {str(data[quant].transform)}"
                        )
        plt.show()
        if interactive:
            input("Press to continue")
            plt.close("all")
