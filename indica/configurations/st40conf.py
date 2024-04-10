# TODO: implement test to check availability of instruments methods and
#       quantities have corresponding DATATYPE

class ST40Conf:
    def __init__(self):

        self.MACHINE_DIMS = ((0.15, 0.85), (-0.75, 0.75))
        self.INSTRUMENT_METHODS = {
            "efit": "get_equilibrium",
            "xrcs": "get_helike_spectroscopy",
            "princeton": "get_charge_exchange",
            "lines": "get_diode_filters",
            "nirh1": "get_interferometry",
            "nirh1_bin": "get_interferometry",
            "smmh1": "get_interferometry",
            "smmh": "get_interferometry",
            "astra": "get_astra",
            "sxr_spd": "get_radiation",
            "sxr_diode_1": "get_diode_filters",
            "sxr_diode_2": "get_diode_filters",
            "sxr_diode_3": "get_diode_filters",
            "sxr_diode_4": "get_diode_filters",
            "sxr_mid1": "get_radiation",
            "sxr_mid2": "get_radiation",
            "sxr_mid3": "get_radiation",
            "sxr_mid4": "get_radiation",
            "sxrc_xy1": "get_radiation",
            "sxrc_xy2": "get_radiation",
            "blom_xy1": "get_radiation",
            "cxff_pi": "get_charge_exchange",
            "cxff_tws_c": "get_charge_exchange",
            "cxqf_tws_c": "get_charge_exchange",
            "pi": "get_spectrometer",
            "tws_c": "get_spectrometer",
            "ts": "get_thomson_scattering",
        }

        self.UIDS_MDS = {
            "xrcs": "sxr",
        }
        self.QUANTITIES_MDS = {
            "efit": {
                "f": ".profiles.psi_norm:f",
                "faxs": ".global:faxs",
                "fbnd": ".global:fbnd",
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
            "xrcs": {
                "int_k": ".te_kw:int_k",
                "int_w": ".te_kw:int_w",
                "int_z": ".te_kw:int_z",
                "int_q": ".te_kw:int_q",
                "int_r": ".te_kw:int_r",
                "int_a": ".te_kw:int_a",
                "int_n3": ".te_n3w:int_n3",
                "int_tot": ".te_n3w:int_tot",
                "te_kw": ".te_kw:te",
                "te_n3w": ".te_n3w:te",
                "ti_w": ".ti_w:ti",
                "ti_z": ".ti_z:ti",
                "ampl_w": ".ti_w:amplitude",
                "spectra": ":intensity",
            },
            "nirh1": {
                "ne": ".line_int:ne",
            },
            "nirh1_bin": {
                "ne": ".line_int:ne",
            },
            "smmh1": {
                "ne": ".line_int:ne",
            },
            "smmh": {
                "ne": ".global:ne_int",
            },
            "lines": {
                "brightness": ":emission",
            },
            "sxr_spd": {
                "brightness": ".profiles:emission",
            },
            "sxr_mid1": {
                "brightness": ".profiles:emission",
            },
            "sxr_mid2": {
                "brightness": ".profiles:emission",
            },
            "sxr_mid3": {
                "brightness": ".profiles:emission",
            },
            "sxr_mid4": {
                "brightness": ".profiles:emission",
            },
            "diode_arrays": {
                "brightness": ".middle_head.filter_4:",
                "location": ".middle_head.geometry:location",
                "direction": ".middle_head.geometry:direction",
            },
            "sxrc_xy1": {
                "brightness": ".profiles:emission",
            },
            "sxrc_xy2": {
                "brightness": ".profiles:emission",
            },
            "blom_xy1": {
                "brightness": ".profiles:emission",
            },
            "sxr_diode_1": {
                "brightness": ".filter_001:signal",
            },
            "sxr_diode_2": {
                "brightness": ".filter_002:signal",
            },
            "sxr_diode_3": {
                "brightness": ".filter_003:signal",
            },
            "sxr_diode_4": {
                "brightness": ".filter_004:signal",
            },
            "cxff_pi": {
                "int": ".profiles:int",
                "ti": ".profiles:ti",
                "vtor": ".profiles:vtor",
                "spectra": ":spectra",
                "fit": ":full_fit",
            },
            "cxff_tws_c": {
                "int": ".profiles:int",
                "ti": ".profiles:ti",
                "vtor": ".profiles:vtor",
                "spectra": ":spectra",
                "fit": ":full_fit",
            },
            "pi": {
                "spectra": ":emission",
            },
            "tws_c": {
                "spectra": ":emission",
            },
            "ts": {
                "ne": ".profiles:ne",
                "te": ".profiles:te",
                "pe": ".profiles:pe",
                "chi2": ".profiles:chi2",
            },
            "astra": {
                "f": ".profiles.psi_norm:fpol",
                "faxs": ".global:faxs",
                "fbnd": ".global:fbnd",
                "ftor": ".profiles.psi_norm:ftor",
                "psi_1d": ".profiles.psi_norm:psi",
                "psi": ".psi2d:psi",
                "volume": ".profiles.psi_norm:volume",
                "area": ".profiles.psi_norm:areat",
                "rmag": ".global:rmag",
                "rgeo": ".global:rgeo",
                "zmag": ".global:zmag",
                "zgeo": ".global:zgeo",
                "rbnd": ".p_boundary:rbnd",
                "zbnd": ".p_boundary:zbnd",
                "wp": ".global:wth",
                "ipla": ".global:ipl",
                "upl": ".global:upl",
                "wth": ".global:wth",
                "wtherm": ".global:wtherm",
                "wfast": ".global:wfast",
                "df": ".global.df",
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
                "rho": ".profiles.astra:rho",  # ASTRA rho-toroidal
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
                "areat": ".profiles.psi_norm:areat",  # Toroidal cross section, m2
                "p": ".profiles.psi_norm:p",
                "pblon": ".profiles.astra:pblon",
                "pbper": ".profiles.astra:pbper",
                "pnb": ".global:pnb",  # W
                "pabs": ".global:pabs",  # W
                "p_oh": ".global:p_oh",  # W
                "q": ".profiles.psi_norm:q",
                "sigmapar": ".profiles.psi_norm:sigmapar",  # 1/(Ohm*m)
                "nn": ".profiles.astra:nn",  # 10^19/m^3
                "niz1": ".profiles.astra:niz1",  # 10^19/m^3
                "niz2": ".profiles.astra:niz2",  # 10^19/m^3
                "niz3": ".profiles.astra:niz3",  # 10^19/m^3
            },
        }
