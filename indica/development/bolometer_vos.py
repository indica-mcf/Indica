from copy import deepcopy
import getpass

from matplotlib import rcParams
import matplotlib.pylab as plt
import numpy as np

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.bolometer_camera import Bolometer
from indica.models.plasma import example_run as example_plasma
from indica.models.plasma import Plasma
from indica.operators import tomo_1D
from indica.readers.read_st40 import ReadST40
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

FIG_PATH = f"/home/{getpass.getuser()}/figures/Indica/bolometer_vos/"

CMAP, COLORS = set_plot_colors()
set_plot_rcparams("profiles")
rcParams.update({"font.size": 16})


def bolo_xy_p23(
    pulse: int = 11336,
    instrument: str = "sxrc_xy1",
    plasma: Plasma = None,
    st40: ReadST40 = None,
    save_fig: bool = False,
):

    if plasma is None:
        plasma = example_plasma(pulse=pulse, calc_power_loss=True)

    if st40 is None:
        st40 = ReadST40(pulse)
        st40(instruments=[instrument])
    quantity = list(st40.raw_data[instrument])[0]
    los_transform = st40.raw_data[instrument][quantity].transform
    los_transform.set_equilibrium(plasma.equilibrium, force=True)
    origin = los_transform.origin
    direction = los_transform.direction

    model = Bolometer(
        instrument,
    )

    # Nominal LOS geometry
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    # Shifted LOS geometry
    los_transform_plus = LineOfSightTransform(
        origin_x=origin[:, 0] + 0.01,
        origin_y=origin[:, 1],
        origin_z=origin[:, 2],
        direction_x=direction[:, 0],
        direction_y=direction[:, 1],
        direction_z=direction[:, 2],
        machine_dimensions=los_transform._machine_dims,
    )
    los_transform_plus.set_equilibrium(plasma.equilibrium)
    model_plus = deepcopy(model)
    model_plus.set_los_transform(los_transform_plus)

    los_transform_minus = LineOfSightTransform(
        origin_x=origin[:, 0] - 0.01,
        origin_y=origin[:, 1],
        origin_z=origin[:, 2],
        direction_x=direction[:, 0],
        direction_y=direction[:, 1],
        direction_z=direction[:, 2],
        machine_dimensions=los_transform._machine_dims,
    )
    los_transform_minus.set_equilibrium(plasma.equilibrium)
    model_minus = deepcopy(model)
    model_minus.set_los_transform(los_transform_minus)

    bckc = model()
    bckc_plus = model_plus()
    bckc_minus = model_minus()

    cols = model.los_transform.plot(orientation="xy")
    for ch in model.los_transform.x1:
        plt.plot(
            model_plus.los_transform.x.sel(channel=ch),
            model_plus.los_transform.y.sel(channel=ch),
            color=cols[ch],
            linewidth=2,
            alpha=0.4,
        )
        plt.plot(
            model_minus.los_transform.x.sel(channel=ch),
            model_minus.los_transform.y.sel(channel=ch),
            color=cols[ch],
            linewidth=2,
            alpha=0.4,
        )
    plt.title("LOS geometry (x,y)")
    save_figure(FIG_PATH, "los_geometry", save_fig=save_fig)

    cols_time = CMAP(np.linspace(0.1, 0.75, len(model.t), dtype=float))
    plt.figure()
    for i, t in enumerate(model.t.values):
        nominal = bckc["brightness"].sel(t=t, method="nearest")
        plus = bckc_plus["brightness"].sel(t=t, method="nearest")
        minus = bckc_minus["brightness"].sel(t=t, method="nearest")
        nominal.plot(label=f"t={t:1.2f} s", color=cols_time[i], alpha=0.5)
        plt.fill_between(nominal.channel, plus, minus, color=cols_time[i], alpha=0.7)

    set_axis_sci()
    plt.xlabel("Channel")
    plt.ylabel("[W/$m^2$]")
    plt.legend()
    plt.title("Measured brightness")
    save_figure(FIG_PATH, "bckc_brightness", save_fig=save_fig)

    # Local emissivity profiles
    plt.figure()
    for i, t in enumerate(model.t.values):
        model.emissivity.sel(t=t).plot(
            label=f"t={t:1.2f} s", color=cols_time[i], alpha=0.8
        )

    set_axis_sci()
    plt.xlabel("rho")
    plt.ylabel("[W/$m^3$]")
    plt.legend()
    plt.title("Local radiated power")
    save_figure(FIG_PATH, "local_radiated_power", save_fig=save_fig)

    data_t0 = bckc["brightness"].isel(t=0).data
    has_data = np.logical_not(np.isnan(data_t0)) & (data_t0 >= 1.0e3)

    rho_equil = model.los_transform.equilibrium.rho.interp(
        t=bckc["brightness"].t, method="nearest"
    )
    input_dict = dict(
        brightness=bckc["brightness"].data,
        dl=model.los_transform.dl,
        t=bckc["brightness"].t.data,
        R=model.los_transform.R.data,
        z=model.los_transform.z.data,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        emissivity=model.emissivity,
        debug=False,
        has_data=has_data,
    )
    # return input_dict

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=0.5)

    tomo()
    plt.ioff()
    tomo.show_reconstruction()

    return plasma, st40
