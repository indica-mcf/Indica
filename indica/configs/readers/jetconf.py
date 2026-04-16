from pathlib import Path

from indica.configs.readers.machineconf import MachineConf


class JETConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
        self.INSTRUMENT_METHODS = {
            "efit": "equilibrium",
            "eftp": "equilibrium",
            "eftf": "equilibrium",
            "hrts": "thomson_scattering",
            "lidr": "thomson_scattering",
            "kg1v": "interferometry",
            "kg4r": "polarimetry",
            "kk3": "cyclotron_emissions",
            "kg10": "density_reflectometer",
            "sxrh": "radiation",
            "sxrv": "radiation",
            "sxrt": "radiation",
            "kb5h": "radiation",
            "kb5v": "radiation",
            "ks3h": "zeff",
            "ks3v": "zeff",
            **{
                "cx{}{}".format(val1, val2): "charge_exchange"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
            **{
                "cx{}{}_zeff".format(val1, val2): "zeff"
                for val1 in ("s", "d", "f", "g", "h")
                for val2 in ("m", "w", "x", "4", "6", "8")
            },
        }
        self.QUANTITIES_PATH = {
            "equilibrium": {
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
            "thomson_scattering": {"z": "z", "ne": "ne", "te": "te"},
            "interferometry": {
                **{"R": "r", "z": "z", "a": "a"},
                **{"LID{}".format(i): "lid{}".format(i) for i in range(1, 9)},
            },
            "polarimetry": {
                **{"R": "r", "z": "z", "a": "a"},
                **{"FAR{}".format(i): "far{}".format(i) for i in range(1, 9)},
            },
            "cyclotron_emissions": {"gen": "gen", "te": "tprf", "R": "cprf"},
            "density_reflectometer": {"R": "r", "z": "z", "ne": "ne"},
            "sxr_radiation": {},
            "radiation": {},
            "zeff": {"zeff_avrg": "zfbr"},
            "charge_exchange": {
                "R": "rpos",
                "z": "pos",
                "ti": "ti",
                "vtor": "angf",
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
        self.SURF_PATH = Path(__file__).parent.parent.parent / "data/surf_los.dat"
