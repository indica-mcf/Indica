import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.transect import TransectCoordinates
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.models.neutral_beam import NeutralBeam
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

import sys

class ChargeExchange(DiagnosticModel):
    """
    Object representing a CXRS diagnostic
    """

    def __init__(
        self,
        name: str,
        element: str = "c",
        instrument_method="get_charge_exchange",
    ):
        self.name = name
        self.element = element
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]
        self.beam = None

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "vtor":
                quantity = quant
                self.bckc[quantity] = self.Vtor_at_channels
                long_name = "Toroidal rotation"
                units = "rad/s"
            elif quant == "ti":
                quantity = quant
                self.bckc[quantity] = self.Ti_at_channels
                long_name = "Ion temperature"
                units = "eV"
            elif quant == "spectra":
                # Placeholder
                continue
            elif quant == "fit":
                # Placeholder
                continue
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            error = xr.full_like(self.bckc[quantity], 0.0)
            stdev = xr.full_like(self.bckc[quantity], 0.0)
            self.bckc[quantity].attrs = {
                "datatype": datatype,
                "transform": self.transect_transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
                "long_name": long_name,
                "units": units,
            }

    def __call__(
        self,
        Ti: DataArray = None,
        Vtor: DataArray = None,
        t: LabeledArray = None,
        calc_rho: bool = False,
        method: str = 'sample',
        run_fidasim: bool = False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ti
            Ion temperature profile (dims = "rho", "t")
        Vtor
            Toroidal rotation profile (dims = "rho", "t")

        Returns
        -------
        Dictionary of back-calculated quantities (identical to abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ti = self.plasma.ion_temperature.interp(t=t)
            Vtor = self.plasma.toroidal_rotation.interp(t=t)
        else:
            if Ti is None or Vtor is None:
                raise ValueError("Give inputs or assign plasma class!")

        if "element" in Vtor.dims:
            Vtor = Vtor.sel(element=self.element)
        if "element" in Ti.dims:
            Ti = Ti.sel(element=self.element)

        self.t = t
        self.Vtor = Vtor
        self.Ti = Ti

        if method == 'sample':
            ti_at_channels = self.transect_transform.map_profile_to_rho(
                Ti, t=t, calc_rho=calc_rho
            )
            vtor_at_channels = self.transect_transform.map_profile_to_rho(
                    Vtor, t=t, calc_rho=calc_rho
            )
        elif method == 'fidasim':

            if self.beam is None:
                raise ValueError("Please assign beam class!")

            # Get rho points
            rho, _ = self.transect_transform.convert_to_rho_theta(t=self.plasma.t)

            # Run fidasim
            fidasim_results = self.run_fidasim(
                run_fidasim=run_fidasim,
            )

            # Coordinates
            channels = np.arange(len(self.transect_transform.x))
            #coords = [
            #    ("channel", channels),
            #    ("t", fidasim_results["time"]),
            #]
            coords = {
                "channel": channels,
                "t": fidasim_results["time"],
                "element": "c",
                "rho_poloidal": rho,
            }

            # Translate results to DataArray
            ti_at_channels = DataArray(
                data=fidasim_results["Ti"],
                coords=coords,
                dims=("channel", "t"),
            )
            ti_at_channels.attrs = {
                "datatype": ("temperature", "ion"),
            }
            ti_err_at_channels = DataArray(
                data=fidasim_results["Ti_err"],
                coords=coords,
                dims=("channel", "t")
            )
            ti_at_channels.attrs["error"] = ti_err_at_channels
            ti_at_channels.attrs["transform"] = self.transect_transform

            vtor_at_channels = DataArray(
                data=fidasim_results["vtor"],
                coords=coords,
                dims=("channel", "t"),
            )
            vtor_at_channels.attrs = {
                "datatype": ("linear_rotation", "ion"),
            }
            vtor_err_at_channels = DataArray(
                data=fidasim_results["vtor_err"],
                coords=coords,
                dims=("channel", "t"),
            )
            vtor_at_channels.attrs["error"] = vtor_err_at_channels
            vtor_at_channels.attrs["transform"] = self.transect_transform

        else:
            raise ValueError(f'No method available for {method}')

        self.Ti_at_channels = ti_at_channels
        self.Vtor_at_channels = vtor_at_channels

        self._build_bckc_dictionary()

        return self.bckc

    def set_beam(
        self,
        beam: NeutralBeam,
    ):

        self.beam = beam

        return

    def run_fidasim(
        self,
        run_fidasim: bool = False,
        quiet: bool = True
    ):

        # Pulse number
        pulse = self.plasma.pulse

        # Build beam configuration
        nbiconfig = {
            "name": self.beam.name,
            "einj": self.beam.energy * 1e-3,
            "pinj": self.beam.power * 1e-6,
            "current_fractions": [
                self.beam.fractions[0],
                self.beam.fractions[1],
                self.beam.fractions[2]
            ],
            "ab": self.beam.amu
        }

        # Add geom_dict and chord_IDs to specconfig
        origin = self.los_transform.origin
        direction = self.los_transform.direction
        chord_ids = [f"M{i + 1}" for i in range(np.shape(direction)[0])]
        geom_dict = dict()
        for i_chord, id in enumerate(chord_ids):
            geom_dict[id] = {}
            geom_dict[id]["origin"] = list(origin[i_chord, :] * 1e2)
            geom_dict[id]["diruvec"] = list(direction[i_chord, :])
        specconfig = {
            "chord_IDs": chord_ids,
            "geom_dict": geom_dict,
            "name": self.name,
            "cross_section_corr": False,
        }

        # Atomic mass of plasma ion
        if self.plasma.main_ion == 'h':
            plasma_ion_amu = 1.00874
        elif self.plasma.main_ion == 'd':
            plasma_ion_amu = 2.014
        else:
            raise ValueError('Plasma ion must be Hydrogen "h" or Deuterium "d"')

        # Times to analyse
        times = self.plasma.t.data
        beam_on = np.ones(len(self.plasma.t.data))

        # Loop
        Ti = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
        Ti_err = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
        vtor = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
        vtor_err = np.zeros((len(specconfig["chord_IDs"]), len(times))) * np.nan
        for i_time, time in enumerate(times):
            if beam_on[i_time]:
                # Extract data from plasma / equilibrium objects
                # profiles
                rho_1d = self.plasma.ion_temperature.coords["rho_poloidal"]
                ion_temperature = self.plasma.ion_temperature.sel(element='c', t=time).values
                electron_temperature = self.plasma.electron_temperature.sel(t=time).values
                electron_density = self.plasma.electron_density.sel(t=time).values
                neutral_density = self.plasma.neutral_density.sel(t=time).values
                toroidal_rotation = self.plasma.toroidal_rotation.sel(element='c', t=time).values
                zeffective = self.plasma.zeff.sum("element").sel(t=time).values

                # magnetic data
                # rho poloidal
                rho_2d = self.plasma.equilibrium.rho.interp(
                    t=time,
                    method="nearest"
                )

                # rho toroidal
                rho_tor, _ = self.plasma.equilibrium.convert_flux_coords(rho_2d, t=time)
                rho_tor = rho_tor.values  # NaN's is this going to be an issue?

                # radius
                R = self.plasma.equilibrium.rho.R.values

                # vertical position
                z = self.plasma.equilibrium.rho.z.values

                # meshgrid
                R_2d, z_2d = np.meshgrid(R, z)

                # Br
                br, _ = self.plasma.equilibrium.Br(
                    self.plasma.equilibrium.rho.R,
                    self.plasma.equilibrium.rho.z,
                    t=time
                )
                br = br.values

                # Bz
                bz, _ = self.plasma.equilibrium.Bz(
                    self.plasma.equilibrium.rho.R,
                    self.plasma.equilibrium.rho.z,
                    t=time
                )
                bz = bz.values

                # Bt, ToDo: returns NaNs!!
                # bt, _ = beam.plasma.equilibrium.Bt(
                #    beam.plasma.equilibrium.rho.R,
                #    beam.plasma.equilibrium.rho.z,
                #    t=time
                # )
                # bt = bt.values

                # Bypass bug -> irod = 2*pi*R * BT / mu0_fiesta;
                irod = 3.0 * 1e6
                bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

                # rho
                rho = rho_2d.values

                plasmaconfig = {
                    "R": R_2d,
                    "z": z_2d,
                    "rho_1d": rho_1d,
                    "rho": rho,
                    "rho_t": rho_tor,
                    "br": br,
                    "bz": bz,
                    "bt": bt,
                    "ti": ion_temperature,
                    "te": electron_temperature,
                    "nn": neutral_density,
                    "ne": electron_density,
                    "omegator": toroidal_rotation,
                    "zeff": zeffective,
                    "plasma_ion_amu": plasma_ion_amu,
                }

                if not quiet:
                    # Print statements
                    print(f'pulse = {pulse}')
                    print(f'time = {time}')
                    print(f'nbiconfig = {nbiconfig}')
                    print(f'specconfig = {specconfig}')
                    print(f'plasmaconfig = {plasmaconfig}')
                    print(f'plasma_ion_amu = {plasma_ion_amu}')
                    print(f'run_fidasim = {run_fidasim}')

                    # Plot psi map
                    plt.figure()
                    plt.contourf(R, z, rho_2d.values)

                    # Plot magnetic fields
                    plt.figure()
                    plt.subplot(131)
                    plt.contourf(
                        self.plasma.equilibrium.rho.R,
                        self.plasma.equilibrium.rho.z,
                        br,
                    )
                    plt.subplot(132)
                    plt.contourf(
                        self.plasma.equilibrium.rho.R,
                        self.plasma.equilibrium.rho.z,
                        bz,
                    )
                    plt.subplot(133)
                    plt.contourf(
                        self.plasma.equilibrium.rho.R,
                        self.plasma.equilibrium.rho.z,
                        bt,
                    )
                    plt.show(block=True)

                # Run TE-fidasim
                path_to_code = '/home/jonathan.wood/git_home/te-fidasim'
                sys.path.append(path_to_code)
                import fidasim_ST40_indica
                results = fidasim_ST40_indica.main(
                    pulse,
                    time,
                    nbiconfig,
                    specconfig,
                    plasmaconfig,
                    num_cores=3,
                    user="jonathan.wood",
                    force_run_fidasim=run_fidasim,
                    save_dir="/home/jonathan.wood/fidasim_output",
                )
                Ti[:, i_time] = results["Ti"]
                Ti_err[:, i_time] = results["Ti_err"]
                vtor[:, i_time] = results["vtor"]
                vtor_err[:, i_time] = results["vtor_err"]

        out = {
            'time': times,
            'Ti': Ti,
            'Ti_err': Ti_err,
            'vtor': vtor,
            'vtor_err': vtor_err,
        }

        return out

    def calculate_neutral_density_along_los(self, debug=True):

        # Neutral density map
        nh = self.beam.nh
        beam_source = self.beam.source
        beam_angle = self.beam.angle

        # Get neutral density along CX LOS chords
        los_x = self.los_transform.x
        los_y = self.los_transform.y
        los_z = self.los_transform.z
        print(f"los_x = {los_x}")

        # Rotate LOS to beam axis
        def rotate(x, y, xo, yo, theta):
            xr = np.cos(theta) * (x - xo) - np.sin(theta) * (y - yo) + xo
            yr = np.sin(theta) * (x - xo) + np.cos(theta) * (y - yo) + yo
            return [xr, yr]

        vec = rotate(
            los_x - beam_source[0],
            los_y - beam_source[1],
            0.0,
            0.0,
            -beam_angle,
        )
        los_x_rot = vec[0]
        los_y_rot = vec[1]
        los_z_rot = los_z

        print(los_x_rot)
        print(los_y_rot)
        print(los_z_rot)
        print(vec)

        # Interpolate along neutral density grid
        excite = 1
        nh_ = nh[0].isel(excitations=excite)
        print(nh_)
        print(nh_.shape)
        #print("aa"**2)

        # Loop over channel
        if debug:
            from matplotlib import pyplot as plt
            import matplotlib.cm as cm

            cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(self.los_transform.x1), dtype=float))

            plt.figure()
            ax1 = plt.axes()

            plt.figure()
            ax2 = plt.axes()

            ax2.contour(
                self.beam.depth_grid,
                self.beam.width_grid,
                nh_[0, 0, 19, :, :], 50)

            plt.figure()
            ax3 = plt.axes()

        nh_along_chord_list: list = []
        for i, i_ch in enumerate(self.los_transform.x1):

            nh_along_chord_beamlet: list = []
            for i_beamlet in range(self.los_transform.beamlets):

                print(f"i_ch = {i_ch}")
                print(los_x_rot)

                nh_along_chord = nh_.interp(
                    depth_grid=los_x_rot.isel(channel=i_ch, beamlet=i_beamlet),
                    width_grid=los_y_rot.isel(channel=i_ch, beamlet=i_beamlet),
                    height_grid=los_z_rot.isel(channel=i_ch, beamlet=i_beamlet)
                )

                print(nh_along_chord.isel(energy=0))
                print(nh_along_chord.isel(energy=0).shape)

                nh_along_chord_beamlet.append(nh_along_chord)

                if debug:
                    ax1.plot(
                        nh_along_chord.isel(energy=0)[0, :],
                        c = cols[i],
                    )

                    ax2.plot(
                        los_x_rot.isel(channel=i_ch, beamlet=i_beamlet),
                        los_y_rot.isel(channel=i_ch, beamlet=i_beamlet),
                        c = cols[i],
                    )
                    ax2.plot(
                        los_x_rot.isel(channel=i_ch, beamlet=i_beamlet)[0],
                        los_y_rot.isel(channel=i_ch, beamlet=i_beamlet)[0],
                        'ko')

                    ax3.plot(
                        los_x.isel(channel=i_ch, beamlet=i_beamlet),
                        los_y.isel(channel=i_ch, beamlet=i_beamlet),
                        c=cols[i],
                    )

            nh_along_chord_list.append(nh_along_chord_beamlet)

        nh_along_chord_array = DataArray(
            nh_along_chord_list,
            coords=[
                ("channel", los_x.channel),
                ("beamlet", los_x.beamlet),
                ("energy", np.arange(0, 3, 1, dtype=int)),
                ("time", nh_.t),
                ("los_position", los_x.los_position)
            ]

        )

        if debug:
            plt.show(block=True)

        self.nh_along_chord = nh_along_chord_array

        return


def pi_transform_example(nchannels: int):
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)

    transect_transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        "pi",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
    )
    return transect_transform


def example_run(
    pulse: int = None,
    diagnostic_name: str = "cxrs",
    plasma=None,
    plot=False,
):
    # TODO: LOS sometimes crossing bad EFIT reconstruction

    if plasma is None:
        plasma = example_plasma(pulse=pulse)

    # Create new interferometers diagnostics
    transect_transform = pi_transform_example(5)
    transect_transform.set_equilibrium(plasma.equilibrium)
    model = ChargeExchange(
        diagnostic_name,
    )
    model.set_transect_transform(transect_transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it].values

        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))

        model.transect_transform.plot(tplot)

        # Plot back-calculated profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.toroidal_rotation.sel(t=t, element=model.element).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Vtor = bckc["vtor"].sel(t=t, method="nearest")
            rho = Vtor.transform.rho.sel(t=t, method="nearest")
            plt.scatter(rho, Vtor, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured toroidal rotation (rad/s)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.ion_temperature.sel(t=t, element=model.element).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Ti = bckc["ti"].sel(t=t, method="nearest")
            rho = Ti.transform.rho.sel(t=t, method="nearest")
            plt.scatter(rho, Ti, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured ion temperature (eV)")
        plt.legend()
        plt.show()

    return plasma, model, bckc


if __name__ == "__main__":
    plt.ioff()
    example_run(plot=True)
    plt.show()
