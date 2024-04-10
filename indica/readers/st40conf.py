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
            "princeton": "spectrom",
            "nirh1": "interferom",
            "nirh1_bin": "interferom",
            "smmh1": "interferom",
            "sxr_diode_1": "sxr",
            "sxr_diode_2": "sxr",
            "sxr_diode_3": "sxr",
            "sxr_diode_4": "sxr",
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
                "ftor": ".profiles.psi_norm:ftor",  # Wb
                # "rmji": ".profiles.psi_norm:rmji",
                # "rmjo": ".profiles.psi_norm:rmjo",
                "psi_1d": ".profiles.psi_norm:psi",
                "psi": ".psi2d:psi",
                # "vjac": ".profiles.psi_norm:vjac",
                # "ajac": ".profiles.psi_norm:ajac",
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
                "elon": ".profiles.astra:elon",  # Elongation profile
                "j_bs": ".profiles.astra:j_bs",  # Bootstrap current density,MA/m2
                "j_nbi": ".profiles.astra:j_nbi",  # NB driven current density,MA/m2
                "j_oh": ".profiles.astra:j_oh",  # Ohmic current density,MA/m2
                "j_rf": ".profiles.astra:j_rf",  # EC driven current density,MA/m2
                "j_tot": ".profiles.astra:j_tot",  # Total current density,MA/m2
                "ne": ".profiles.astra:ne",  # Electron density, 10^19 m^-3
                "ni": ".profiles.astra:ni",  # Main ion density, 10^19 m^-3
                "nf": ".profiles.astra:nf",  # Main ion density, 10^19 m^-3
                "n_d": ".profiles.astra:n_d",  # Deuterium density,10E19/m3
                "n_t": ".profiles.astra:n_t",  # Tritium density	,10E19/m3
                "omega_tor": ".profiles.astra:omega_tor",  # Toroidal rot. freq., 1/s
                "qe": ".profiles.astra:qe",  # electron power flux, MW
                "qi": ".profiles.astra:qi",  # ion power flux, MW
                "qn": ".profiles.astra:qn",  # total electron flux, 10^19/s
                "qnbe": ".profiles.astra:qnbe",  # Beam power density to electrons, MW/m3
                "qnbi": ".profiles.astra:qnbi",  # Beam power density to ions, MW/m3
                "q_oh": ".profiles.astra:q_oh",  # Ohmic heating power profile, MW/m3
                "q_rf": ".profiles.astra:q_rf",  # RF power density to electron,MW/m3
                "rho": ".profiles.astra:rho",  # ASTRA rho-toroidal
                "rmid": ".profiles.astra:rmid",  # Centre of flux surfaces, m
                "rminor": ".profiles.astra:rminor",  # minor radius, m
                "sbm": ".profiles.astra:sbm",  # Particle source from beam, 10^19/m^3/s
                "spel": ".profiles.astra:spel",  # Particle source from pellets, 10^19/m^3/s
                "stot": ".profiles.astra:stot",  # Total electron source,10^19/s/m3
                "swall": ".profiles.astra:swall",  # Wall neutrals source, 10^19/m^3/s
                "te": ".profiles.astra:te",  # Electron temperature, keV
                "ti": ".profiles.astra:ti",  # Ion temperature, keV
                "tri": ".profiles.astra:tri",  # Triangularity (up/down symmetrized) profile
                "t_d": ".profiles.astra:t_d",  # Deuterium temperature,keV
                "t_t": ".profiles.astra:t_t",  # Tritium temperature,keV
                "zeff": ".profiles.astra:zeff",  # Effective ion charge
                "areat": ".profiles.psi_norm:areat",  # Toroidal cross section,m2
                "p": ".profiles.psi_norm:p",  # PRESSURE(PSI_NORM)
                "pblon": ".profiles.astra:pblon",  # PRESSURE(PSI_NORM)
                "pbper": ".profiles.astra:pbper",  # PRESSURE(PSI_NORM)
                "pnb": ".global:pnb",  # Injected NBI power, W
                "pabs": ".global:pabs",  # Absorber NBI power, W
                "p_oh": ".global:p_oh",  # Absorber NBI power, W
                "q": ".profiles.psi_norm:q",  # Q_PROFILE(PSI_NORM)
                "sigmapar": ".profiles.psi_norm:sigmapar",  # Paral. conduct.,1/(Ohm*m)
                "nn": ".profiles.astra:nn",  # Thermal neutral density, 10^19/m^3
                "niz1": ".profiles.astra:niz1",  # Impurity density, 10^19/m^3
                "niz2": ".profiles.astra:niz2",  # Impurity density, 10^19/m^3
                "niz3": ".profiles.astra:niz3",  # Impurity density, 10^19/m^3
            },
        }
