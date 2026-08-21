from indica.configs.readers.machineconf import MachineConf


class MASTUConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((0.1, 2.0), (-2.2, 2.2))
        self.INSTRUMENT_METHODS = {
            "efit": "equilibrium",
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
        }
