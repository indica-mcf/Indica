# Create trees from ppac standard utility tools

import sys

from MDSplus import Connection
import numpy as np
import standard_utility as util


BDA_NODES = {
    "TIME": ("NUMERIC", "time vector of optimisation, s"),
    "INPUT": {
        "BURN_FRAC": ("NUMERIC", "Burn in fraction for chains"),
        "ITER": ("NUMERIC", "Maximum iterations of optimiser"),
        "PARAM_NAMES": ("TEXT", "Names of parameters optimised"),
        "OPT_QUANTITY": ("TEXT", "Names of quantities optimised"),
        "MODEL_KWARGS": ("TEXT", "Model key word arguments"),
        # "OPT_KWARGS": ("TEXT", "Optimiser key word arguments"),
        "PULSE": ("NUMERIC", "Pulse number"),
        "TSTART": ("NUMERIC", "Start of time vector, s"),
        "TEND": ("NUMERIC", "End of time vector, s"),
        "DT": ("NUMERIC", "Distance between time points, s"),
        "IMPURITIES": ("TEXT", "Names of impurity elements"),
        "MAIN_ION": ("TEXT", "Name of main ion"),
    },
    "METADATA": {
        "GITCOMMIT": ("TEXT", "Commit ID used for run"),
        "USER": ("TEXT", "Username of owner"),
        "EQUIL": ("TEXT", "Equilibrium used"),
    },
    "PROFILES": {
        "PSI_NORM": {
            "RHOP": ("NUMERIC", "Radial vector, Sqrt of normalised poloidal flux"),
            "RHOT": ("SIGNAL", "Radial vector, toroidal flux"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
            "TIZ1": ("SIGNAL", "Ion temperature of impurity Z1, eV"),
            "TIZ2": ("SIGNAL", "Ion temperature of impurity Z2, eV"),
            "TIZ3": ("SIGNAL", "Ion temperature of impurity Z3, eV"),
            "NIZ1": ("SIGNAL", "Density of impurity Z1, m^-3"),
            "NIZ2": ("SIGNAL", "Density of impurity Z2, m^-3"),
            "NIZ3": ("SIGNAL", "Density of impurity Z3, m^-3"),
            "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
            "NFAST": ("SIGNAL", "Density of fast ion, m^-3"),
            "ZI": ("SIGNAL", "Average charge of main ion, "),
            "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
            "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
            "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
            "ZEFF": ("SIGNAL", "Effective charge, "),
            "P": ("SIGNAL", "Pressure,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "NE_ERR": ("SIGNAL", "Electron density error, m^-3"),
            "NI_ERR": ("SIGNAL", "Ion density error, m^-3"),
            "TE_ERR": ("SIGNAL", "Electron temperature error, eV"),
            "TI_ERR": ("SIGNAL", "Ion temperature of main ion error, eV"),
            "TIZ1_ERR": ("SIGNAL", "Ion temperature of impurity Z1 error, eV"),
            "TIZ2_ERR": ("SIGNAL", "Ion temperature of impurity Z2 error, eV"),
            "TIZ3_ERR": ("SIGNAL", "Ion temperature of impurity Z3 error, eV"),
            "NIZ1_ERR": ("SIGNAL", "Density of impurity Z1 error, m^-3"),
            "NIZ2_ERR": ("SIGNAL", "Density of impurity Z2 error, m^-3"),
            "NIZ3_ERR": ("SIGNAL", "Density of impurity Z3 error, m^-3"),
            "NNEUTR_ERR": ("SIGNAL", "Density of neutral main ion error, m^-3"),
            "NFAST_ERR": ("SIGNAL", "Density of fast ion error, m^-3"),
            "ZI_ERR": ("SIGNAL", "Average charge of main ion error, "),
            "ZIM1_ERR": ("SIGNAL", "Average charge of impurity IMP1 error, "),
            "ZIM2_ERR": ("SIGNAL", "Average charge of impurity IMP2 error, "),
            "ZIM3_ERR": ("SIGNAL", "Averagnodese charge of impurity IMP3 error, "),
            "ZEFF_ERR": ("SIGNAL", "Effective charge error, "),
        },
        "R_MIDPLANE": {
            "RPOS": ("NUMERIC", "Major radius position of measurement, m"),
            "ZPOS": ("NUMERIC", "Z position of measurement, m"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
            "TIZ1": ("SIGNAL", "Ion temperature of impurity Z1, eV"),
            "TIZ2": ("SIGNAL", "Ion temperature of impurity Z2, eV"),
            "TIZ3": ("SIGNAL", "Ion temperature of impurity Z3, eV"),
            "NIZ1": ("SIGNAL", "Density of impurity Z1, m^-3"),
            "NIZ2": ("SIGNAL", "Density of impurity Z2, m^-3"),
            "NIZ3": ("SIGNAL", "Density of impurity Z3, m^-3"),
            "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
            "NFAST": ("SIGNAL", "Density of fast ion, m^-3"),
            "ZI": ("SIGNAL", "Average charge of main ion, "),
            "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
            "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
            "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
            "ZEFF": ("SIGNAL", "Effective charge, "),
            "P": ("SIGNAL", "Pressure,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "NE_ERR": ("SIGNAL", "Electron density error, m^-3"),
            "NI_ERR": ("SIGNAL", "Ion density error, m^-3"),
            "TE_ERR": ("SIGNAL", "Electron temperature error, eV"),
            "TI_ERR": ("SIGNAL", "Ion temperature of main ion error, eV"),
            "TIZ1_ERR": ("SIGNAL", "Ion temperature of impurity Z1 error, eV"),
            "TIZ2_ERR": ("SIGNAL", "Ion temperature of impurity Z2 error, eV"),
            "TIZ3_ERR": ("SIGNAL", "Ion temperature of impurity Z3 error, eV"),
            "NIZ1_ERR": ("SIGNAL", "Density of impurity Z1 error, m^-3"),
            "NIZ2_ERR": ("SIGNAL", "Density of impurity Z2 error, m^-3"),
            "NIZ3_ERR": ("SIGNAL", "Density of impurity Z3 error, m^-3"),
            "NNEUTR_ERR": ("SIGNAL", "Density of neutral main ion error, m^-3"),
            "NFAST_ERR": ("SIGNAL", "Density of fast ion error, m^-3"),
            "ZI_ERR": ("SIGNAL", "Average charge of main ion error, "),
            "ZIM1_ERR": ("SIGNAL", "Average charge of impurity IMP1 error, "),
            "ZIM2_ERR": ("SIGNAL", "Average charge of impurity IMP2 error, "),
            "ZIM3_ERR": ("SIGNAL", "Average charge of impurity IMP3 error, "),
            "ZEFF_ERR": ("SIGNAL", "Effective charge error, "),
        },
    },
    "PROFILE_STAT": {
        "SAMPLE_IDX": ("NUMERIC", "Index of the optimisation samples"),
        "RHOP": ("NUMERIC", "Radial vector, Sqrt of normalised poloidal flux"),
        "NE": ("SIGNAL", "Electron density, m^-3"),
        "NI": ("SIGNAL", "Ion density, m^-3"),
        "TE": ("SIGNAL", "Electron temperature, eV"),
        "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
        "TIZ1": ("SIGNAL", "Ion temperature of impurity IMP1, eV"),
        "TIZ2": ("SIGNAL", "Ion temperature of impurity IMP2, eV"),
        "TIZ3": ("SIGNAL", "Ion temperature of impurity IMP3, eV"),
        "NIZ1": ("SIGNAL", "Density of impurity IMP1, m^-3"),
        "NIZ2": ("SIGNAL", "Density of impurity IMP2, m^-3"),
        "NIZ3": ("SIGNAL", "Density of impurity IMP3, m^-3"),
        "NNEUTR": ("SIGNAL", "Density of neutral main ion, m^-3"),
        "NFAST": ("SIGNAL", "Density of fast ions, m^-3"),
        "ZI": ("SIGNAL", "Average charge of main ion, "),
        "ZIM1": ("SIGNAL", "Average charge of impurity IMP1, "),
        "ZIM2": ("SIGNAL", "Average charge of impurity IMP2, "),
        "ZIM3": ("SIGNAL", "Average charge of impurity IMP3, "),
        "ZEFF": ("SIGNAL", "Effective charge, "),
        "P": ("SIGNAL", "Pressure,Pa"),
        "VOLUME": ("SIGNAL", "Volume inside magnetic surface, m^3"),
    },
    "GLOBAL": {
        "NE0": ("SIGNAL", "Central electron density, m^-3"),
        "NI0": ("SIGNAL", "Central ion density, m^-3"),
        "TE0": ("SIGNAL", "Central electron temperature, eV"),
        "TI0": ("SIGNAL", "Central ion temperature of main ion, eV"),
        "TI0Z1": ("SIGNAL", "Central ion temperature of impurity Z1, eV"),
        "TI0Z2": ("SIGNAL", "Central ion temperature of impurity Z2, eV"),
        "TI0Z3": ("SIGNAL", "Central ion temperature of impurity Z3, eV"),
        "NI0Z1": ("SIGNAL", "Central density of impurity Z1, m^-3"),
        "NI0Z2": ("SIGNAL", "Central density of impurity Z2, m^-3"),
        "NI0Z3": ("SIGNAL", "Central density of impurity Z3, m^-3"),
        "NE0_ERR": ("SIGNAL", "Central electron density error, m^-3"),
        "NI0_ERR": ("SIGNAL", "Central ion density error, m^-3"),
        "TE0_ERR": ("SIGNAL", "Central electron temperature error, eV"),
        "TI0_ERR": ("SIGNAL", "Central ion temperature of main ion error, eV"),
        "TI0Z1_ERR": ("SIGNAL", "Central ion temperature of impurity Z1 error, eV"),
        "TI0Z2_ERR": ("SIGNAL", "Central ion temperature of impurity Z2 error, eV"),
        "TI0Z3_ERR": ("SIGNAL", "Central ion temperature of impurity Z3 error, eV"),
        "NI0Z1_ERR": ("SIGNAL", "Central density of impurity Z1 error, m^-3"),
        "NI0Z2_ERR": ("SIGNAL", "Central density of impurity Z2 error, m^-3"),
        "NI0Z3_ERR": ("SIGNAL", "Central density of impurity Z3 error, m^-3"),
        "WP": ("SIGNAL", "Stored energy, J"),
        "WTH": ("SIGNAL", "Thermal component of stored energy, J"),
        "PTOT": ("SIGNAL", "Total pressure, Pa"),
        "PTH": ("SIGNAL", "Thermal pressure, Pa"),
        "WP_ERR": ("SIGNAL", "Stored energy error, J"),
        "WTH_ERR": ("SIGNAL", "Thermal component of stored energy error, J"),
        "PTOT_ERR": ("SIGNAL", "Total pressure error, Pa"),
        "PTH_ERR": ("SIGNAL", "Thermal pressure error, Pa"),
    },
    "PHANTOMS": {
        "FLAG": ("TEXT", "True if phantoms used"),
        "RHO_POLOIDAL": (
            "NUMERIC",
            "Radial vector, Sqrt of normalised poloidal flux",
        ),
        "NE": ("SIGNAL", "Electron density, m^-3"),
        "NI": ("SIGNAL", "Ion density, m^-3"),
        "TE": ("SIGNAL", "Electron temperature, eV"),
        "TI": ("SIGNAL", "Ion temperature of main ion, eV"),
        "TIZ1": ("SIGNAL", "Ion temperature of impurity Z1 , eV"),
        "TIZ2": ("SIGNAL", "Ion temperature of impurity Z2, eV"),
        "TIZ3": ("SIGNAL", "Ion temperature of impurity Z3, eV"),
        "NIZ1": ("SIGNAL", "Impurity density of Z1, m^-3"),
        "NIZ2": ("SIGNAL", "Impurity density of Z2, m^-3"),
        "NIZ3": ("SIGNAL", "Impurity density of Z3, m^-3"),
    },
    "OPTIMISATION": {
        "ACCEPT_FRAC": ("NUMERIC", "Fraction of samples accepted by optimiser"),
        "AUTO_CORR": ("NUMERIC", "Auto-correlation"),
        "POST_SAMPLE": ("NUMERIC", "Posterior probability samples"),
        "PRIOR_SAMPLE": ("NUMERIC", "Prior samples"),
        "GELMAN_RUBIN": ("NUMERIC", "Gelmin-Rubin convergence diagnostic"),
    },
}


DIAGNOSTIC_QUANTITY = [
    "DIAGNOSTIC1.QUANTITY1",
    "DIAGNOSTIC1.QUANTITY2",
    "DIAGNOSTIC2.QUANTITY1",
]


def create_nodes(
    pulse_to_write=43000000,
    run="RUN01",
    best=True,
    diagnostic_quantities=DIAGNOSTIC_QUANTITY,
    mode="EDIT",
):
    bda_nodes = BDA_NODES
    quant_list = [
        item.upper().split(".") for item in diagnostic_quantities
    ]  # replace OPTIMISED_QUANTITY
    diag_names = list(set([item[0] for item in quant_list]))

    diag_nodes = {
        diag_name: {
            quantity[1]: ("SIGNAL", f"measured {quantity[1]} from {quantity[0]}")
            for quantity in quant_list
            if quantity[0] == diag_name
        }
        for diag_name in diag_names
    }

    nodes = {
        "RUN": ("TEXT", "RUN used for diagnostic"),
        "USAGE": ("TEXT", "Quantity used in analysis"),
        "PULSE": ("NUMERIC", "Pulse used for diagnostic"),
    }

    workflow_nodes = {diag_name: nodes for diag_name in diag_names}

    model_nodes = {
        diag_name: {
            quantity[1]: ("SIGNAL", f"modelled {quantity[1]} from {quantity[0]}")
            for quantity in quant_list
            if quantity[0] == diag_name
        }
        for diag_name in diag_names
    }
    model_nodes["SAMPLE_IDX"] = ("NUMERIC", "Index of the optimisation samples")
    bda_nodes["MODEL_DATA"] = model_nodes
    bda_nodes["DIAG_DATA"] = diag_nodes
    bda_nodes["INPUT"]["WORKFLOW"] = workflow_nodes

    tree = "BDA"
    script_name = ""
    script_info = ""

    node_info = util.GetNodeInformation(
        script=None,
        node_information_type="json",
        run_name=run,
        run_info=run,
        script_name=script_name,
        script_info=script_info,
        root_node=None,
        tree=tree,
        pulse_number=pulse_to_write,
        base_node_to_read=None,
        node_information_file=bda_nodes,
    ).get()

    util.StandardNodeCreation(
        pulse_number=pulse_to_write,
        dict_node_info=node_info,
        mode=mode,
        name_of_BEST="BEST",  # name of the structure linked to BEST
        link_BEST_to_run=best,
    )
    return node_info


def check_to_overwrite_run(
    pulseNo,
    which_run,
):
    # Checker function to see if data already exists in a run
    IP_address_smaug = "smaug"
    conn = Connection(IP_address_smaug)
    conn.openTree("BDA", pulseNo)

    temp = conn.get("BDA." + which_run + ".TIME").data()
    conn.closeAllTrees()

    overwrite_flag = True
    if isinstance(temp, np.ndarray):
        print(f"Data already Exists in pulseNo = {pulseNo}, which_run = {which_run}")
        print("User prompt...")
        question = (
            f"    Scheduled to overwrite pulseNo {pulseNo}, {which_run}"
            f"\n    Do you want to overwrite {which_run}? (y/n)"
        )
        overwrite_flag = query_yes_no(question)
    return overwrite_flag


def does_tree_exist(
    pulse,
):
    IP_address_smaug = "smaug"
    conn = Connection(IP_address_smaug)

    try:
        conn.openTree("BDA", pulse)
        conn.closeAllTrees()
        return True
    except Exception:
        return False


def query_yes_no(
    question,
):
    valid = {"yes": True, "y": True, "no": False, "n": False}
    while True:
        sys.stdout.write(question)
        choice = input().lower()
        if choice == "":
            return False
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


if __name__ == "__main__":

    pulse = 43000000
    run = "RUN01"

    tree_exists = does_tree_exist(pulse)
    if tree_exists:
        mode = "EDIT"
    else:
        mode = "NEW"
    create_nodes(pulse_to_write=pulse, mode=mode, run=run)
