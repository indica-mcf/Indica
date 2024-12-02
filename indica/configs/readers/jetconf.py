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
            "ks3h": "get_bremsstrahlung_spectroscopy",
            "ks3v": "get_bremsstrahlung_spectroscopy",
            "sxrh": "get_radiation",
            "sxrv": "get_radiation",
            "sxrt": "get_radiation",
            "kb5h": "get_radiation",
            "kb5v": "get_radiation",
            "kg10": "get_density_reflectometer",
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
            "get_thomson_scattering": {key: key for key in ["ne", "te"]},
            "get_cyclotron_emissions": {key: key for key in ["te"]},
            "get_reflectometer": {key: key for key in ["ne"]},
            "get_radiation": {key: "brightness" for key in ["h", "v", "t"]},
            "get_bremsstrahlung_spectroscopy": {"zefh": "zeff", "zefv": "zeff"},
            "get_charge_exchange": {"ti": "ti", "angf": "vtor", "conc": "conc"},
        }
