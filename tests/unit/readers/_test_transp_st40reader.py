from indica.readers import ST40Reader

PULSE = 40000042
TSTART = 0
TEND = 20
REVISION = "J10"
TREE = "TRANSP_TEST"

INSTRUMENTS: list = ["transp_test"]


def run_reader_get_methods(
    reader: ST40Reader,
    instrument: str,
):
    print(instrument)
    data = reader.get("", instrument, revision=REVISION, return_dataarrays=True)
    return data


def test_reader_get_methods(return_dataarrays=False, verbose=False):
    reader = ST40Reader(
        PULSE,
        TSTART,
        TEND,
        verbose=verbose,
        tree=TREE,
    )
    for instrument in INSTRUMENTS:
        data = run_reader_get_methods(reader, instrument)
        assert type(data) == dict
        assert len(data) > 0

        for key in data.keys():
            print(key+": ",data[key].shape)

        import matplotlib.pyplot as plt
        plotted_quantities=["volume","te","ne"]

        for quantity_to_plot in plotted_quantities:
            dataslice=data[quantity_to_plot]
            time_dim, radial_dim = dataslice.dims
            t_index=0
            quantity_t=dataslice.isel({time_dim:t_index})
            rho=dataslice["rhot"]



            # 4. Plot Te(rho) at the chosen time
            plt.figure()
            plt.plot(rho, quantity_t.values)
            plt.xlabel("rhot")
            plt.ylabel(quantity_to_plot)
            plt.title(f"{quantity_to_plot} vs rhot at {time_dim} index = {t_index}")

            # 5. Save instead of show
            plt.savefig(f"{quantity_to_plot}_vs_rho_t0.png", dpi=150, bbox_inches="tight")
            plt.close()






test_reader_get_methods()
