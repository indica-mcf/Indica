from indica.configs.readers.machineconf import MachineConf


class ST40Conf(MachineConf):
    def __init__(self):
        self.MACHINE_DIMS = ((0.15, 0.85), (-0.75, 0.75))
        self.INSTRUMENT_METHODS = {
            "efit": "get_equilibrium",
            "xrcs": "get_helike_spectroscopy",
            "pi": "get_spectrometer",
            "tws_c": "get_spectrometer",
            "cxff_pi": "get_charge_exchange",
            "cxff_tws_c": "get_charge_exchange",
            "cxff_tws_b": "get_charge_exchange",
            "cxqf_tws_c": "get_charge_exchange",
            "lines": "get_diode_filters",
            "smmh": "get_interferometry",
            "nirh": "get_interferometry",
            "ts": "get_thomson_scattering",
            "sxr_spd": "get_radiation",
            "sxrc_xy1": "get_radiation",
            "sxrc_xy2": "get_radiation",
            "blom_xy1": "get_radiation",
            "astra": "get_astra",
            "ppts": "get_profile_fits",
            "zeff_brems": "get_zeff",
        }
        self.QUANTITIES_PATH = {
            "get_equilibrium": {
                "t": ":time",
                "psin": ".profiles.psi_norm:xpsn",
                "R": ".psi2d:rgrid",
                "z": ".psi2d:zgrid",
                "f": ".profiles.psi_norm:f",
                "psi_axis": ".global:faxs",
                "psi_boundary": ".global:fbnd",
                "ftor": ".profiles.psi_norm:ftor",
                "rmji": ".profiles.psi_norm:rmji",
                "rmjo": ".profiles.psi_norm:rmjo",
                "psi": ".psi2d:psi",
                "vjac": ".profiles.psi_norm:vjac",
                "ajac": ".profiles.psi_norm:ajac",
                "rmag": ".global:rmag",
                "rgeo": ".global:rgeo",
                "rbnd": ".p_boundary:rbnd",
                "zmag": ".global:zmag",
                "zbnd": ".p_boundary:zbnd",
                "ipla": ".constraints.ip:cvalue",
                "wp": ".virial:wp",
                "df": ".constraints.df:cvalue",
            },
            "get_helike_spectroscopy": {
                "t": ":time",
                "wavelength": ":wavelen",
                "location": ".geometry:location",
                "direction": ".geometry:direction",
                "ti_w": ".global:ti_w",
                "ti_z": ".global:ti_z",
                "te_n3w": ".global:te_n3w",
                "te_kw": ".global:te_kw",
                "spectra_raw": ":intens",
                "spectra": ":spec_rad",
                "int_w": ".global:int_w",
                "int_k": ".global:int_k",
                "int_tot": ".global:int_tot",
                "int_n3": ".global:int_n3",
                "background": ".global:back_avg",
            },
            "get_interferometry": {
                "t": ":time",
                "location": ".geometry:location",
                "direction": ".geometry:direction",
                "ne": ".global:ne_int",
            },
            "get_diode_filters": {
                "t": ":time",
                "label": ":label",
                "location": ".geometry:location",
                "direction": ".geometry:direction",
                "brightness": ":data",
            },
            "get_radiation": {
                "t": ":time",
                "location": ".geometry:location",
                "direction": ".geometry:direction",
                "brightness": ".profiles:emission",
            },
            "get_charge_exchange": {
                "t": ":time",
                "wavelength": ":wavelen",
                "x": ":x",
                "y": ":y",
                "z": ":z",
                "R": ":R",
                "int": ".profiles:int",
                "ti": ".profiles:ti",
                "vtor": ".profiles:vtor",
                "spectra": ":spectra",
                "fit": ":full_fit",
            },
            "get_spectrometer": {
                "t": ":time",
                "location": ".geometry:location",
                "direction": ".geometry:direction",
                "wavelength": ":wavelen",
                "spectra": ":emission",
            },
            "get_thomson_scattering": {
                "t": ":time",
                "x": ":x",
                "y": ":y",
                "z": ":z",
                "R": ":R",
                "ne": ".profiles:ne",
                "te": ".profiles:te",
                "pe": ".profiles:pe",
                "chi2": ".profiles:chi2",
            },
            "get_profile_fits": {
                "t": ":time",
                "rhop_fit": ".profiles.psi_norm:rhop",
                "R_fit": ".profiles.r_midplane:rpos",
                "rhop": ".profiles.inputs:rhop",
                "R": ".profiles.inputs:rpos",
                "z": ".profiles.inputs:zpos",
                "ne_rhop": ".profiles.psi_norm:ne",
                "te_rhop": ".profiles.psi_norm:te",
                "pe_rhop": ".profiles.psi_norm:pe",
                "ne_R": ".profiles.r_midplane:ne",
                "te_R": ".profiles.r_midplane:te",
                "pe_R": ".profiles.r_midplane:pe",
                "ne_data": ".profiles.inputs:ne",
                "te_data": ".profiles.inputs:te",
                "pe_data": ".profiles.inputs:pe",
                "R_shift": ".global:rshift",
            },
            "get_zeff": {
                "t": ":time",
                "rhop": ".profiles.psi_norm:rhop",
                "R_shift": ".global:rshift",
                "zeff_avrg": ".global:zeff",
                "zeff_hi": ".global:zeff_hi",
                "zeff_low": ".global:zeff_low",
                "zeff": ".profiles.psi_norm:zeff",
            },
            "get_astra": {
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
