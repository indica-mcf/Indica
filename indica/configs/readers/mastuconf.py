from indica.configs.readers.machineconf import MachineConf


class MASTUConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((0.1, 2.0), (-2.2, 2.2))
        self.INSTRUMENT_METHODS = {
            "magnetics_efit": "equilibrium",
            "cel3": "cx_spectrometer",
            "cel4b": "cx_spectrometer",
            "cel4c": "cx_spectrometer",
        }
        self.QUANTITIES_PATH = {
            "equilibrium": {
                "t": "time",
                "R": "output/profiles2D/r",
                "z": "output/profiles2D/z",
                "rmag": "output/globalParameters/magneticAxis/R",
                "zmag": "output/globalParameters/magneticAxis/Z",
                "psi_axis": "output/globalParameters/psiAxis",
                "psi_boundary": "output/globalParameters/psiBoundary",
                "rbnd": "output/SeparatrixGeometry/rBoundary",
                "zbnd": "output/SeparatrixGeometry/zBoundary",
                "f": "output/fluxFunctionProfiles/rBphi",
                "ftor": "output/fluxFunctionProfiles/toroidalFlux",
                "xpsin": "output/fluxFunctionProfiles/normalizedPoloidalFlux",
                "rmji": "output/fluxFunctionsProfiles/rInboard",
                "rmjo": "output/fluxFunctionsProfiles/rOutboard",
                "psi": "output/profiles2D/psiNorm",
            },
            "cx_spectrometer": {
                "t": "ss/time",
                "channel": "ss/spectrometer_fibre",
                "wavelength": "ss/wavelength",
                "spectrometer_counts": "ss/counts",
                "instrument_function": "ss/instrument_function",
                "sensitivity": "ss/sensitivity",
                "pvb_background": "ss/pvb/scaled_bg_counts",
                "bnb_background": "ss/bnb/scaled_bg_counts",
                "location": "ss/location",
                "direction": "ss/direction",
            },
        }
