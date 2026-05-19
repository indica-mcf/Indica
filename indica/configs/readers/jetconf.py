from pathlib import Path

from indica.configs.readers.machineconf import MachineConf


class JETConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
        self.INSTRUMENT_METHODS = {
            "efit": "get_equilibrium",
            "eftp": "get_equilibrium",
            "eftf": "get_equilibrium",
            "hrts": "get_thomson_scattering",
            "lidr": "get_thomson_scattering",
            "kg1v": "get_interferometry",
            "kg4r": "get_polarimetry",
            "kk3": "get_cyclotron_emissions",
            "kg10": "get_density_reflectometer",
            "sxrh": "get_radiation",
            "sxrv": "get_radiation",
            "sxrt": "get_radiation",
            "kb5h": "get_radiation",
            "kb5v": "get_radiation",
            "ks3h": "get_zeff",
            "ks3v": "get_zeff",
            "ks3h_bash": "get_spectrometer",
            "ks3v_basv": "get_spectrometer",
            **{
                "cx{}{}".format(val1, val2): "get_charge_exchange"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
            **{
                "cx{}{}_zeff".format(val1, val2): "get_zeff"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
            **{
                "cx{}{}_base".format(val1, val2): "get_spectrometer"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
            **{f"t{i:>03}": "get_thomson_scattering" for i in range(1000)},
        }
        self.QUANTITIES_PATH = {
            "get_equilibrium": {
                "R": "psir",
                "z": "psiz",
                "rgeo": "rgeo",
                "rmag": "rmag",
                "zmag": "zmag",
                "psi_axis": "faxs",
                "psi_boundary": "fbnd",
                "wp": "wp",
                "rbnd": "rbnd",
                "zbnd": "zbnd",
                "f": "f",
                "ftor": "ftor",
                "rmji": "rmji",
                "rmjo": "rmjo",
                "vjac": "vjac",
                "ajac": "ajac",
                "psi": "psi",
            },
            "get_thomson_scattering": {"z": "z", "ne": "ne", "te": "te"},
            "get_interferometry": {
                **{"R": "r", "z": "z", "a": "a"},
                **{"LID{}".format(i): "lid{}".format(i) for i in range(1, 9)},
            },
            "get_polarimetry": {
                **{"R": "r", "z": "z", "a": "a"},
                **{"FAR{}".format(i): "far{}".format(i) for i in range(1, 9)},
            },
            "get_cyclotron_emissions": {"gen": "gen", "te": "tprf", "R": "cprf"},
            "get_density_reflectometer": {"R": "r", "z": "z", "ne": "ne"},
            "get_sxr_radiation": {},
            "get_radiation": {},
            "get_spectrometer": {},
            "get_zeff": {"zeff_avrg": "zfbr"},
            "get_charge_exchange": {
                "R": "rpos",
                "z": "pos",
                "ti": "ti",
                "angf": "afcr",  # angf
                "vtor": "afcr",  # angf -> vtor in reader step
                "conc": "conc",
                "dens": "dens",
                "zeff_avrg": "zfbr",
            },
        }
        self._BREMSSTRAHLUNG_LOS = {
            "ks3": "edg7",
        }
        self._RADIATION_RANGES = {
            "sxr/h": 17,
            "sxr/t": 35,
            "sxr/v": 35,
            "bolo/kb5h": 24,
            "bolo/kb5v": 32,
        }
        self._KK3_RANGE = (1, 96)
        self.SURF_PATH = Path(__file__).parent.parent.parent / "data/surf_los.dat"
