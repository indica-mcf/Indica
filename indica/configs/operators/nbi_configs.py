import numpy as np


# Beam geometry: RFX beamline configuration used to build the FIDASIM beam grid.
def get_rfx_geo():
    """ """
    rfx = {}

    rfx["name"] = "rfx"
    rfx["shape"] = 2
    rfx["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    rfx["src"] = 100 * np.array([-2.0199, -2.6323, 0.0])
    tangency = 100 * np.array([0.2985, -0.2955, 0.0])
    rfx["axis"] = (tangency - rfx["src"]) / np.linalg.norm(tangency - rfx["src"])
    rfx["widy"] = 17.2
    rfx["widz"] = 17.2
    # rfx["widy"] = 1. # narrow beams
    # rfx["widz"] = 1. # narrow beams
    rfx["divy"] = np.array([0.014, 0.014, 0.014])
    rfx["divz"] = np.array([0.014, 0.014, 0.014])
    # rfx["divy"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    # rfx["divz"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    rfx["focy"] = 160.0
    rfx["focz"] = 160.0
    # rfx["focy"] = 300.0 # narrow beams
    # rfx["focz"] = 300.0 # narrow beams
    rfx["naperture"] = 0

    # Pencil-like
    # rfx["widy"] = 1.0
    # rfx["widz"] = 1.0
    # rfx["divy"] = np.array([0.001,0.001,0.001])
    # rfx["divz"] = np.array([0.001,0.001,0.001])

    return rfx


# Beam geometry: HNBI beamline configuration used to build the FIDASIM beam grid.
def get_hnbi_geo():
    """ """
    hnbi = {}

    hnbi["name"] = "hnbi"
    hnbi["shape"] = 2
    hnbi["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    hnbi["src"] = 100 * np.array([3.322, 3.945, 0.0])
    tangency = 100 * np.array([-0.2985, 0.2955, 0.0])
    hnbi["axis"] = (tangency - hnbi["src"]) / np.linalg.norm(tangency - hnbi["src"])
    # hnbi["widy"] = 25.0
    # hnbi["widz"] = 25.0
    hnbi["widy"] = 12.5  # numbers from Jari on 29/06/23 via teams
    hnbi["widz"] = 12.5
    hnbi["divy"] = np.array([0.014, 0.014, 0.014])
    hnbi["divz"] = np.array([0.014, 0.014, 0.014])
    # hnbi["focy"] = 420.0
    # hnbi["focz"] = 420.0
    hnbi["focy"] = 355.0
    hnbi["focz"] = 355.0
    hnbi["naperture"] = 0
    # inputs["pinj"] = 0.6
    # inputs["einj"] = 55.0
    # inputs["current_fractions"] = np.array([0.64,0.25,0.11])

    # Pencil-like
    # hnbi["widy"] = 1.0
    # hnbi["widz"] = 1.0
    # hnbi["divy"] = np.array([0.001,0.001,0.001])
    # hnbi["divz"] = np.array([0.001,0.001,0.001])

    return hnbi
