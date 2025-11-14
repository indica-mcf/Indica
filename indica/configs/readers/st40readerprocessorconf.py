import numpy as np


class ST40ReaderProcessorConf:
    def __init__(self):

        self.filter_values = {
            "cxff_pi": {"ti": (0, np.inf), "vtor": (0, np.inf)},
            "cxff_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
            "cxff_tws_b": {"ti": (0, np.inf), "vtor": (0, np.inf)},
            "cxqf_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
            "xrcs": {
                "ti_w": (0, np.inf),
                "ti_z": (0, np.inf),
                "te_kw": (0, np.inf),
                "te_n3w": (0, np.inf),
                "spectra": (0, np.inf),
                "spectra_raw": (0, np.inf),
            },
            "brems": {"brightness": (0, np.inf)},
            "halpha": {"brightness": (0, np.inf)},
            "sxr_spd": {"brightness": (0, np.inf)},
            "sxr_diode_1": {"brightness": (0, np.inf)},
            "sxr_camera_4": {"brightness": (0, np.inf)},
            "sxrc_xy1": {"brightness": (0, np.inf)},
            "sxrc_xy2": {"brightness": (0, np.inf)},
            "blom_xy1": {"brightness": (0, np.inf)},
            "smmh": {"ne": (0, np.inf)},
            "ts": {"te": (0, np.inf), "ne": (0, np.inf)},
            "ppts": {
                "te_rho": (1, np.inf),
                "ne_rho": (1, np.inf),
                "pe_rho": (1, np.inf),
                "te_R": (1, np.inf),
                "ne_R": (1, np.inf),
                "pe_R": (1, np.inf),
            },
            "pi": {"spectra": (0, np.inf)},
            "tws_c": {"spectra": (0, np.inf)},
        }

        self.filter_coordinates = {
            "cxff_pi": {
                "ti": ("channel", (0, np.inf)),
                "vtor": ("channel", (0, np.inf)),
            },
            "cxff_tws_c": {
                "ti": ("channel", (0, np.inf)),
                "vtor": ("channel", (0, np.inf)),
            },
            "cxff_tws_b": {
                "ti": ("channel", (0, np.inf)),
                "vtor": ("channel", (0, np.inf)),
            },
            "xrcs": {
                "spectra": ("wavelength", (0.0, np.inf)),
                "spectra_raw": ("wavelength", (0.0, np.inf)),
            },
            "ts": {"te": ("channel", (0, np.inf)), "ne": ("channel", (0, np.inf))},
        }
