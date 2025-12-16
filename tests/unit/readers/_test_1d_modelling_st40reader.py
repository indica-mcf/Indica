from indica.readers import ST40Reader

TSTART = 0
TEND = 20

INSTRUMENTS = {
    "astra": {"pulse": 13013666, "revision": "RUN602"},
    "transp_test": {"pulse": 40000042, "revision": "J10"},
    "metis": {"pulse": 40011890, "revision": "RUN01"},
}


def _test_reader_get_methods(
    instrument: str,
    return_dataarrays: bool = True,
    verbose: bool = False,
    plot: bool = False,
):
    pulse = INSTRUMENTS[instrument]["pulse"]
    revision = INSTRUMENTS[instrument]["revision"]
    reader = ST40Reader(
        pulse,
        TSTART,
        TEND,
        verbose=verbose,
        tree=instrument,
    )

    data = reader.get(
        "", instrument, revision=revision, return_dataarrays=return_dataarrays
    )

    if verbose:
        for key in data.keys():
            print(key + ": ", data[key].shape)

    if not return_dataarrays:
        return data

    if plot:
        import matplotlib.pyplot as plt

        plotted_quantities = ["te", "ne"]

        for quantity_to_plot in plotted_quantities:
            dataslice = data[quantity_to_plot]
            time_dim, radial_dim = dataslice.dims
            t_index = 0
            quantity_t = dataslice.isel({time_dim: t_index})
            rho = dataslice["rhot"]

            # 4. Plot Te(rho) at the chosen time
            plt.figure()
            plt.plot(rho, quantity_t.values)
            plt.xlabel("rhot")
            plt.ylabel(quantity_to_plot)
            plt.title(f"{quantity_to_plot} vs rhot at {time_dim} index = {t_index}")

            # 5. Save instead of show
            plt.savefig(
                f"{quantity_to_plot}_vs_rho_t0.png", dpi=150, bbox_inches="tight"
            )
            plt.show()

    return data


def _test_astra_reader(verbose: bool = False, plot: bool = False):
    return _test_reader_get_methods("astra")


def _test_transp_reader(verbose: bool = False, plot: bool = False):
    return _test_reader_get_methods("transp")


def _test_metis_reader(verbose: bool = False, plot: bool = False):
    return _test_reader_get_methods("metis")
