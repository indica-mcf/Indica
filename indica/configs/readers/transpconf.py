from indica.configs.readers.machineconf import MachineConf


class TRANSPConf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((0.15, 0.85), (-0.8, 0.8))
        self.INSTRUMENT_METHODS = {
            "efit": "get_equilibrium",
            "transp": "get_transp"


        }
        self.QUANTITIES_PATH = {
            #Todo: remove the astra specific things
              "get_transp": {
                "ne": ".profiles.astra:ne",  # 10^19 m^-3


            },

            "get_transp2": {
                "psi_axis": ".global:faxs",
                "psi_boundary": ".global:fbnd",
                "rmag": ".global:rmag",
                "rgeo": ".global:rgeo",
                "zmag": ".global:zmag",
                "zgeo": ".global:zgeo",
                "wp": ".global:wth",
                "ipla": ".global:ipl",
                "upl": ".global:upl",
                "wth": ".global:wth",
                "wtherm": ".global:wtherm",
                "wfast": ".global:wfast",
                "df": ".global.df",
                "pnb": ".global:pnb",  # W
                "pabs": ".global:pabs",  # W
                "p_oh": ".global:p_oh",  # W
                "q": ".profiles.psi_norm:q",
                "f": ".profiles.psi_norm:fpol",
                "ftor": ".profiles.psi_norm:ftor",
                "psi_1d": ".profiles.psi_norm:psi",
                "p": ".profiles.psi_norm:p",
                "volume": ".profiles.psi_norm:volume",
                "area": ".profiles.psi_norm:areat",
                "sigmapar": ".profiles.psi_norm:sigmapar",  # 1/(Ohm*m)
                "psi": ".psi2d:psi",
                "rbnd": ".p_boundary:rbnd",
                "zbnd": ".p_boundary:zbnd",
                "elon": ".profiles.astra:elon",  #
                "j_bs": ".profiles.astra:j_bs",  # MA/m2
                "j_nbi": ".profiles.astra:j_nbi",  # MA/m2
                "j_oh": ".profiles.astra:j_oh",  # MA/m2
                "j_rf": ".profiles.astra:j_rf",  # MA/m2
                "j_tot": ".profiles.astra:j_tot",  # MA/m2
                "ne": ".profiles.astra:ne",  # 10^19 m^-3
                "ni": ".profiles.astra:ni",  # 10^19 m^-3
                "nf": ".profiles.astra:nf",  # 10^19 m^-3
                "n_d": ".profiles.astra:n_d",  # 10E19/m3
                "n_t": ".profiles.astra:n_t",  # 10E19/m3
                "omega_tor": ".profiles.astra:omega_tor",  # 1/s
                "qe": ".profiles.astra:qe",  # MW
                "qi": ".profiles.astra:qi",  # MW
                "qn": ".profiles.astra:qn",  # 10^19/s
                "qnbe": ".profiles.astra:qnbe",  # MW/m3
                "qnbi": ".profiles.astra:qnbi",  # MW/m3
                "q_oh": ".profiles.astra:q_oh",  # MW/m3
                "q_rf": ".profiles.astra:q_rf",  # MW/m3
                "rhot": ".profiles.astra:rho",  # ASTRA rho-toroidal
                "rmid": ".profiles.astra:rmid",  # Centre of flux surfaces, m
                "rminor": ".profiles.astra:rminor",  # minor radius, m
                "sbm": ".profiles.astra:sbm",  # 10^19/m^3/s
                "spel": ".profiles.astra:spel",  # 10^19/m^3/s
                "stot": ".profiles.astra:stot",  # 10^19/s/m3
                "swall": ".profiles.astra:swall",  # 10^19/m^3/s
                "te": ".profiles.astra:te",  # keV
                "ti": ".profiles.astra:ti",  # keV
                "tri": ".profiles.astra:tri",  # Triangularity (up/down symmetrized)
                "t_d": ".profiles.astra:t_d",  # keV
                "t_t": ".profiles.astra:t_t",  # keV
                "zeff": ".profiles.astra:zeff",
                "pblon": ".profiles.astra:pblon",
                "pbper": ".profiles.astra:pbper",
                "nn": ".profiles.astra:nn",  # 10^19/m^3
                "niz1": ".profiles.astra:niz1",  # 10^19/m^3
                "niz2": ".profiles.astra:niz2",  # 10^19/m^3
                "niz3": ".profiles.astra:niz3",  # 10^19/m^3
            },
        }
