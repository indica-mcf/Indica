from typing import Tuple

import numpy as np

from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run


class NeutralBeam(DiagnosticModel):
    def __init__(
        self,
        name: str = 'rfx',
        element: str = "h",
        amu: float = 2.014,
        energy: float = 25.0 * 1e3,
        power: float = 500.0 * 1e3,
        fractions: tuple = (0.7, 0.1, 0.2, 0.0),
        div_x: float = 14 * 1e-3,
        div_y: float = 14 * 1e-3,
        width_x: float = 0.025,
        width_y: float = 0.025,
    ):
        """Set beam parameters, initialisation"""
        self.name = name  # Beam name
        self.element = element  # Element
        self.amu = amu  # Atomic mass
        self.energy = energy  # Beam energy (eV)
        self.power = power  # Beam power (W)
        self.fractions = fractions  # Beam fractions (%)
        self.div_x = div_x  # Beam divergence in x (rad)
        self.div_y = div_y  # Beam divergence in y (rad)
        self.width_x = width_x  # Beam 1/e width in x (metres)
        self.width_y = width_y  # Beam 1/e width in y (metres)

        # Set Attenuator
        self.attenuator = None

        # Print statements for debugging
        print(f"Beam = {self.name}")
        print(f"Energy = {self.energy} electron-volts")
        print(f"Power = {self.power} watts")
        print(f"Fractions (full, 1/2, 1/3, imp) = {self.fractions} %")
        print(f"divergence (x, y) = ({self.div_x},{self.div_y}) rad")
        print(f"width (x, y) = ({self.width_x},{self.width_y}) metres")

    def _build_bckc_dictionary(self):
        print('Im not sure what this does?')
        return

    def __call__(
            self,
            beam_on: np.ndarray,
            which_code: str = 'FIDASIM',
            which_spectrometer: str = 'Princeton',
    ):
        print('Hello, you called?')
        print(f"You've selected {which_code}")

        # Here's the plasma object
        print(self.plasma)

        # Run the beam code
        if which_code.lower() == 'fidasim':

            self.run_FIDASIM_ST40(
                beam_on,
                which_spectrometer,
            )

        else:

            raise ValueError(f'No available beam code called {which_code}')

        return

    def run_BBNBI(self):
        print("Add code to run BBNBI")

    def run_FIDASIM_ST40(
            self,
            beam_on: np.ndarray,
            which_spectrometer: str,
            path_to_code='/home/jonathan.wood/git_home/te-fidasim',
            force_run_fidasim=True,
    ):
        # Import package - ToDo: move into the Indica
        import sys
        sys.path.append(path_to_code)
        import prepare_fidasim_ST40

        # Times to analyse
        times = self.plasma.t.data
        pulse = self.plasma.pulse
        if pulse == None:
            pulse = 0

        print(f'pulse = {pulse}')
        print(f'times = {times}')
        print(f'beam_on = {beam_on}')

        # Build beam configuration
        nbiconfig = {
            "name": self.name,
            "einj": self.energy * 1e-3,
            "pinj": self.power * 1e-6,
            "current_fractions": [self.fractions[0], self.fractions[1], self.fractions[2]],
            "ab": self.amu
        }

        # Build spectrometer configuration
        specconfig = {
            "name": which_spectrometer,
            "chord_IDs": ["M3", "M4", "M5", "M6", "M7", "M8"],
            "cross_section_corr": True
        }

        print(f'nbiconfig = {nbiconfig}')
        print(f'specconfig = {specconfig}')

        # Atomic mass of plasma ion
        if self.plasma.main_ion == 'h':
            plasma_ion_amu = 1.00874
        elif self.plasma.main_ion == 'd':
            plasma_ion_amu = 2.014
        else:
            raise ValueError('Plasma ion must be Hydrogen "h" or Deuterium "d"')


        # Run Fidasim
        for i_time, time in enumerate(times):
            if beam_on[i_time]:
                # rho poloidal
                rho_2d = self.plasma.equilibrium.rho.interp(
                    t=time,
                    method="nearest"
                )

                # ion temperature
                ion_temperature = self.plasma.ion_temperature.sel(element='c').interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                ion_temperature[np.isnan(ion_temperature)] = 0.0

                # electron temperature
                electron_temperature = self.plasma.electron_temperature.interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                electron_temperature[np.isnan(electron_temperature)] = 0.0

                # electron density
                electron_density = self.plasma.electron_density.interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                electron_density[np.isnan(electron_density)] = 0.0

                # neutral density
                neutral_density = self.plasma.neutral_density.interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                neutral_density[np.isnan(neutral_density)] = 0.0

                # toroidal rotation
                toroidal_rotation = self.plasma.toroidal_rotation.sel(element='c').interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                toroidal_rotation[np.isnan(toroidal_rotation)] = 0.0

                # z-effective
                zeffective = self.plasma.zeff.sum("element").interp(
                    t=time,
                    rho_poloidal=rho_2d
                ).values
                zeffective[np.isnan(zeffective)] = 0.0

                # radius
                R = self.plasma.equilibrium.rho.coords["R"].values

                # vertical position
                z = self.plasma.equilibrium.rho.coords["z"].values

                # meshgrid
                R_2d, z_2d = np.meshgrid(R, z)

                # Br
                br, _ = self.plasma.equilibrium.Br(
                    self.plasma.equilibrium.rho.coords["R"],
                    self.plasma.equilibrium.rho.coords["z"],
                    t=time
                )
                br = br.values

                # Bz
                bz, _ = self.plasma.equilibrium.Bz(
                    self.plasma.equilibrium.rho.coords["R"],
                    self.plasma.equilibrium.rho.coords["z"],
                    t=time
                )
                bz = bz.values

                # Bt, ToDo: returns NaNs!!
                bt, _ = self.plasma.equilibrium.Bt(
                    self.plasma.equilibrium.rho.coords["R"],
                    self.plasma.equilibrium.rho.coords["z"],
                    t=time
                )
                bt = bt.values

                #from matplotlib import pyplot as plt
                #plt.figure()
                #plt.contourf(br)
                #plt.show(block=True)

                # Bypass bug -> irod = 2*pi*R * BT / mu0_fiesta;
                irod = 3.0*1e6
                bt = irod * (4*np.pi * 1e-7) / (2*np.pi * R_2d)

                # rho
                rho = rho_2d.values

                # fidasim grid nR x nZ
                plasmaconfig = {
                    "R": np.transpose(R_2d, (1, 0)),
                    "z": np.transpose(z_2d, (1, 0)),
                    "rho": np.transpose(rho, (1, 0)),
                    "br": np.transpose(br, (1, 0)),
                    "bz": np.transpose(bz, (1, 0)),
                    "bt": np.transpose(bt, (1, 0)),
                    "ti": np.transpose(ion_temperature, (1, 0)),
                    "te": np.transpose(electron_temperature, (1, 0)),
                    "nn": np.transpose(neutral_density, (1, 0)),
                    "ne": np.transpose(electron_density, (1, 0)),
                    "omegator": np.transpose(toroidal_rotation, (1, 0)),
                    "zeff": np.transpose(zeffective, (1, 0)),
                    "plasma_ion_amu": plasma_ion_amu,
                }

                print(self.plasma.toroidal_rotation)
                print(f'ion_temperature = {np.shape(ion_temperature)}')
                print(f'electron_temperature = {np.shape(electron_temperature)}')
                print(f'electron_density = {np.shape(electron_density)}')
                print(f'neutral_density = {np.shape(neutral_density)}')
                print(f'toroidal_rotation = {np.shape(toroidal_rotation)}')
                print(f'zeffective = {zeffective}')
                print(f'R = {R}')
                print(f'z = {z}')
                print(f'Br = {br}')
                print(f'Bt = {bt}')
                print(' ')

                print('Run FIDASIM here')
                prepare_fidasim_ST40.main(
                    pulse,
                    time,
                    nbiconfig,
                    specconfig,
                    plasmaconfig,
                    force_run_fidasim=force_run_fidasim,
                )

                #prepare_fidasim_ST40.main(
                #    shot_number=pulse,
                #    run=run_name,
                #    spec="Princeton",
                #    beam=self.name,
                #    custom_geqdsk=geqdsk,
                #    custom_time=time,
                #    force_run_fidasim=run_fidasim,
                #)

    #def run_FIDASIM_ST40_old(
    #        self,
    #        pulse: int,
    #        run_name: str,
    #        times: list,
    #        path_to_code='/home/jonathan.wood/git_home/te-fidasim'
    #):
    #    print("Add code to run FIDASIM")
    #    import sys
    #    sys.path.append(path_to_code)
    #    import prepare_fidasim_ST40

    #    # Inputs for #10009
    #    geqdsk = 'input/ST40_10009_EFIT_BEST_57p15ms.geqdsk'
    #    run_fidasim = True

    #    # Run Fidasim
    #    for time in times:
    #        prepare_fidasim_ST40.main(
    #            shot_number=pulse,
    #            run=run_name,
    #            spec="Princeton",
    #            beam=self.name,
    #            custom_geqdsk=geqdsk,
    #            custom_time=time,
    #            force_run_fidasim=run_fidasim,
    #        )

    def gaussian_beam_representation(self, nx=101, ny=101, nz=51):
        print("Add code to generate Gaussian beam")

        # Get distance from LineOfSightTransform
        dist = self.transform.distance("los_position", 0, self.transform.x2[0], 0)
        distance = dist[-1]
        print(f'distance = {distance}')

        const = 1.0
        x_ba = np.linspace(-0.2, 0.2, nx, dtype=float)
        y_ba = np.linspace(-0.2, 0.2, ny, dtype=float)
        ell = np.linspace(0.0, 1.0, nz, dtype=float)
        z_ba = distance * ell

        wx0 = 0.05
        wy0 = 0.05
        divx = 7*1e-3
        divy = 10*1e-3
        wx1 = wx0 + distance*np.tan(divx)
        wy1 = wy0 + distance*np.tan(divy)

        wx = np.linspace(wx0, wx1, nz, dtype=float)
        wy = np.linspace(wy0, wy1, nz, dtype=float)

        # Attenuation factor
        attenuation = np.linspace(1.0, 0.0, nz, dtype=float)

        # Meshgrid x, y
        x_2d, y_2d = np.meshgrid(x_ba, y_ba)
        print(np.shape(x_2d))

        # Pseudo-neutral beam density
        n_b = np.zeros((nz, ny, nx), dtype=float)
        for i_ell in range(nz):
            n_b[i_ell, :, :] = const * attenuation[i_ell] * np.exp(-(x_2d**2 / wx[i_ell]**2) - (y_2d**2 / wy[i_ell]**2)) / (wx[i_ell] * wy[i_ell])


        # Debug plotting
        if True:
            y_index = np.argmin(np.abs(y_ba))
            ell_0 = 0.0
            ell_1 = 0.4
            ell_2 = 0.8
            ell_0_index = np.argmin(np.abs(ell - ell_0))
            ell_1_index = np.argmin(np.abs(ell - ell_1))
            ell_2_index = np.argmin(np.abs(ell - ell_2))

            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(131)
            plt.contourf(x_ba, y_ba, n_b[ell_0_index, :, :])
            plt.title(f'ell = {ell_0}')
            plt.subplot(132)
            plt.contourf(x_ba, y_ba, n_b[ell_1_index, :, :])
            plt.title(f'ell = {ell_1}')
            plt.subplot(133)
            plt.contourf(x_ba, y_ba, n_b[ell_2_index, :, :])
            plt.title(f'ell = {ell_2}')
            plt.tight_layout()


            plt.figure()
            plt.plot(x_ba, n_b[ell_0_index, y_index, :], label=f'ell = {ell_0}')
            plt.plot(x_ba, n_b[ell_1_index, y_index, :], label=f'ell = {ell_1}')
            plt.plot(x_ba, n_b[ell_2_index, y_index, :], label=f'ell = {ell_2}')
            plt.legend()
            plt.tight_layout()

            plt.show(block=True)

        # Tait-Bryan angles for converting between coordinate systems
        alpha = -0.7893507119841984
        beta = 0.0
        gamma = 0.0
        self.Rot = self.Tait_Bryan_rotate(alpha, beta, gamma)

        # Assign data to class
        self.x_b = x_ba
        self.y_b = y_ba
        self.z_b = ell
        self.n_b = n_b

    def Tait_Bryan_rotate(self, alpha, beta, gamma):
        """
            https://en.wikipedia.org/wiki/Davenport_chained_rotations
            Section: Tait-Bryan chained rotations
        """

        alpha_yaw = np.array((
            np.array((np.cos(alpha), -1. * np.sin(alpha), 0)),
            np.array((np.sin(alpha), np.cos(alpha), 0)),
            np.array((0, 0, 1)),
        ))
        beta_pitch = np.array((
            np.array((np.cos(beta), 0, np.sin(beta))),
            np.array((0, 1, 0)),
            np.array((-1. * np.sin(beta), 0, np.cos(beta))),
        ))
        gamma_roll = np.array((
            np.array((1, 0, 0)),
            np.array((0, np.cos(gamma), -1. * np.sin(gamma))),
            np.array((0, np.sin(gamma), np.cos(gamma))),
        ))

        M = np.matmul(gamma_roll, np.matmul(beta_pitch, alpha_yaw))

        return M

    def beam_velocity(self):
        return 4.38 * 1e5 * np.sqrt(self.energy * 1e-3 / float(self.amu))


if __name__ == '__main__':
    print('Welcome!')

    # Beam data

    # Generate neutral beam model
    beam = NeutralBeam()

    # Set plasma
    plasma = example_run()
    beam.set_plasma(plasma)

    # Call beam function
    out = beam()
