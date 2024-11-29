from pathlib import Path

from indica.configs.readers.machineconf import MachineConf


class JETConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
        self.INSTRUMENT_METHODS = {
            "efit": "get_equilibrium",
            "eftp": "get_equilibrium",
            "hrts": "get_thomson_scattering",
            "lidr": "get_thomson_scattering",
            "kk3": "get_cyclotron_emissions",
            "kg10": "get_density_reflectometer",
            "sxrh": "get_radiation",
            "sxrv": "get_radiation",
            "sxrt": "get_radiation",
            "kb5h": "get_radiation",
            "kb5v": "get_radiation",
            "ks3h": "get_zeff",
            "ks3v": "get_zeff",
            **{
                "cx{}{}".format(val1, val2): "get_charge_exchange"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
        }
        self.QUANTITIES_PATH = {
            "get_equilibrium": {
                key: key
                for key in [
                    "f",
                    "faxs",
                    "fbnd",
                    "ftor",
                    "rmji",
                    "rmjo",
                    "psi",
                    "vjac",
                    "ajac",
                    "rmag",
                    "rgeo",
                    "rbnd",
                    "zmag",
                    "zbnd",
                    "wp",
                ]
            },
            "get_thomson_scattering": {
                "z": "z",
                "ne": "ne",
                "te": "te",
            },
            "get_cyclotron_emissions": {"te": "te"},
            "get_reflectometer": {"ne": "ne"},
            "get_sxr_radiation": {},
            "get_radiation": {},
            "get_zeff": {"zeff": "zefv"},
            "get_charge_exchange": {
                "R": "rpos",
                "z": "pos",
                "ti": "ti",
                "angf": "vtor",
                "conc": "conc",
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
        self.SURF_PATH = Path(__file__).parent.parent / "data/surf_los.dat"
