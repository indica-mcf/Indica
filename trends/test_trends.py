import trends.trends_database as trends
import trends.analyse_trends as analyse_trends
import trends.plot_trends as plot_trends

PULSE_START = 9770
PULSE_END = 9781
PULSE_ADD = 9790

def test_reload(pulse_start=None, pulse_end=None):
    if pulse_start is None:
        pulse_start = PULSE_START
    if pulse_end is None:
        pulse_end = PULSE_END

    try:
        st40_trends = trends.Database(
            pulse_start=pulse_start, pulse_end=pulse_end, reload=True,
        )
    except FileNotFoundError:
        _ = test_read_and_write(pulse_start=pulse_start, pulse_end=pulse_end,)
        st40_trends = trends.Database(
            pulse_start=pulse_start, pulse_end=pulse_end, reload=True,
        )

    return st40_trends


def test_read_and_write(pulse_start=None, pulse_end=None, set_info=False):
    if pulse_start is None:
        pulse_start = PULSE_START
    if pulse_end is None:
        pulse_end = PULSE_END

    # Initialize class
    st40_trends = trends.Database(
        pulse_start=pulse_start, pulse_end=pulse_end, set_info=set_info
    )

    # Read all data and save to class attributes
    st40_trends()

    # Write information and data to file
    trends.write_database(st40_trends)

    return st40_trends


def test_add_pulses(pulse_start=None, pulse_end=None):
    if pulse_start is None:
        pulse_start = PULSE_START
    if pulse_end is None:
        pulse_end = PULSE_END

    st40_trends = test_reload(pulse_start=pulse_start, pulse_end=pulse_end)

    # Add pulses to database
    st40_trends.add_pulses(pulse_add)

    # Write information and data to file
    trends.write_database(st40_trends)

    # TODO: check whether data is there and as you expect it to be

    return st40_trends


def test_analyse_database(pulse_start=None, pulse_end=None):
    if pulse_start is None:
        pulse_start = PULSE_START
    if pulse_end is None:
        pulse_end = PULSE_END

    st40_trends = test_reload(pulse_start=pulse_start, pulse_end=pulse_end)

    st40_trends = analyse_trends.analyse_database(st40_trends)

    return st40_trends


def test_plot_database(pulse_start=None, pulse_end=None):
    if pulse_start is None:
        pulse_start = PULSE_START
    if pulse_end is None:
        pulse_end = PULSE_END

    st40_trends = test_reload(pulse_start=pulse_start, pulse_end=pulse_end)

    st40_trends = analyse_trends.analyse_database(st40_trends)

    plot_trends.plot(st40_trends)

    return st40_trends


def run_workflow(pulse_start=8207, pulse_end=10046, set_info=False, write=False):
    """
    Run workflow to build Trends database from scratch
    """
    st40_trends = trends.Database(
        pulse_start=pulse_start, pulse_end=pulse_end, set_info=set_info,
    )
    st40_trends()

    if write:
        trends.write_database(st40_trends)

    return st40_trends
