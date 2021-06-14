from copy import deepcopy

import matplotlib.pylab as plt
from hda.atomdat import fractional_abundance
from hda.atomdat import get_atomdat
from hda.atomdat import atomdat_files
from hda.forward_models import Spectrometer
from hda.hdaadas import ADASReader

import pickle

plt.ion()


def fz_table(dump=False, load=False):
    path = "/Users/lavoro/Work/Python/Indica_github/Indica/"
    filename = "fz_data.pkl"

    if load is True:
        fz = pickle.load(open(path + filename, "rb"))
        return fz

    reader = ADASReader()

    files = atomdat_files("")
    elements = files.keys()

    fz = {}
    for elem in elements:
        # Read atomic data
        _, atomdat = get_atomdat(reader, elem, charge="")

        # Interpolate on electron density and drop coordinate
        for k in atomdat.keys():
            atomdat[k] = (
                atomdat[k]
                .interp(electron_density=5.0e19, method="nearest")
                .drop_vars(["electron_density"])
            )

        # Calculate fractional abundance and cooling factor
        fz_tmp = fractional_abundance(atomdat["scd"], atomdat["acd"], element=elem)
        fz[elem] = {
            "values": fz_tmp.values,
            "ion_charges": fz_tmp.ion_charges.values,
            "electron_temperature": fz_tmp.electron_temperature.values,
        }

    if dump is True:
        pickle.dump(fz, open(path + filename, "wb"))

    return fz


def pec_table(dump=False, load=False):

    path = "/Users/lavoro/Work/Python/Indica_github/Indica/"
    filename = "c5_pec_data.pkl"

    if load is True:
        pec = pickle.load(open(path + filename, "rb"))
        return fz

    reader = ADASReader()
    _, atomdat = get_atomdat(
        reader,
        "c",
        "5",
        transition="n=8-n=7",
        wavelength=5292.7,
    )

    pec = spec.atomdat["pec"].interp(electron_density=5.0e19, method="nearest")
    pec = pec.swap_dims({"index": "type"})
    pec_excit = pec.sel(type="excit")
    pec_recom = pec.sel(type="recom")

    pec = {
        "c5/5292.7": {
            "electron_temperature": pec_excit.electron_temperature.values,
            "excit": pec_excit.values,
            "recom": pec_recom.values,
        }
    }

    if dump is True:
        pickle.dump(pec, open(path + filename, "wb"))

    return pec
