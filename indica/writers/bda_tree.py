import sys
from unittest.mock import MagicMock

from MDSplus import Connection
import numpy as np

try:
    import standard_utility as util
except ImportError:
    util = MagicMock
    print("\n ** StandardUtility not installed \n **")

BDA_NODES = {
    "TIME": ("NUMERIC", "Time vector of optimisation"),
    "ELEMENT": ("TEXT", "Element names of ion species"),
    "INPUT": {
        "SETTINGS": ("TEXT", "Settings in config file"),
        "GIT_ID": ("TEXT", "Commit ID used for run"),
        "USER": ("TEXT", "Username of script runner"),
        "DATETIME": ("TEXT", "UTC datetime code was run"),
        "WORKFLOW": {},
    },
    "PROFILES": {
        "PSI_NORM": {
            "RHOP": (
                "NUMERIC",
                "Rho Poloidal - Square root of normalised poloidal flux",
            ),
            "RHOT": (
                "SIGNAL",
                "Rho Toroidal - Square root of normalised toroidal flux",
            ),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface"),
            "NE": ("SIGNAL", "Electron density"),
            "NI": ("SIGNAL", "Ion density"),
            "TE": ("SIGNAL", "Electron temperature"),
            "TI": ("SIGNAL", "Ion temperature"),
            "NNEUTR": ("SIGNAL", "Thermal neutral density"),
            "NFAST": ("SIGNAL", "Fast ion density"),
            "ZEFF": ("SIGNAL", "Effective charge"),
            "MEANZ": ("SIGNAL", "Average ionic charge"),
            "PTH": ("SIGNAL", "Thermal particle pressure"),
            "PFAST": ("SIGNAL", "Fast particle pressure"),
            "P": ("SIGNAL", "Total pressure"),
            "NE_ERR": ("SIGNAL", "Electron density error"),
            "NI_ERR": ("SIGNAL", "Ion density error"),
            "TE_ERR": ("SIGNAL", "Electron temperature error"),
            "TI_ERR": ("SIGNAL", "Ion temperature error"),
            "NNEUTR_ERR": ("SIGNAL", "Thermal neutral density error"),
            "NFAST_ERR": ("SIGNAL", "Fast ion density error"),
            "ZEFF_ERR": ("SIGNAL", "Effective charge error"),
            "MEANZ_ERR": ("SIGNAL", "Average ionic charge error"),
            "PTH_ERR": ("SIGNAL", "Thermal particle pressure error"),
            "PFAST_ERR": ("SIGNAL", "Fast particle pressure error"),
            "P_ERR": ("SIGNAL", "Total pressure error"),
        },
        "R_MIDPLANE": {
            "RPOS": ("NUMERIC", "Major radius position of profile"),
            "ZPOS": ("NUMERIC", "Z position of profile"),
            "NE": ("SIGNAL", "Electron density"),
            "NI": ("SIGNAL", "Ion density"),
            "TE": ("SIGNAL", "Electron temperature"),
            "TI": ("SIGNAL", "Ion temperature"),
            "NNEUTR": ("SIGNAL", "Thermal neutral density"),
            "NFAST": ("SIGNAL", "Fast ion density"),
            "MEANZ": ("SIGNAL", "Average ionic charge"),
            "ZEFF": ("SIGNAL", "Effective charge"),
            "PTH": ("SIGNAL", "Thermal particle pressure"),
            "PFAST": ("SIGNAL", "Fast particle pressure"),
            "P": ("SIGNAL", "Total pressure"),
            "NE_ERR": ("SIGNAL", "Electron density error"),
            "NI_ERR": ("SIGNAL", "Ion density error"),
            "TE_ERR": ("SIGNAL", "Electron temperature error"),
            "TI_ERR": ("SIGNAL", "Ion temperature error"),
            "NNEUTR_ERR": ("SIGNAL", "Thermal neutral density error"),
            "NFAST_ERR": ("SIGNAL", "Fast ion density error"),
            "ZEFF_ERR": ("SIGNAL", "Effective charge error"),
            "MEANZ_ERR": ("SIGNAL", "Average ionic charge error"),
            "PTH_ERR": ("SIGNAL", "Thermal particle pressure error"),
            "PFAST_ERR": ("SIGNAL", "Fast particle pressure error"),
            "P_ERR": ("SIGNAL", "Total pressure error"),
        },
    },
    "PROFILE_STAT": {
        "SAMPLE_IDX": ("NUMERIC", "Index of the optimisation samples"),
        "RHOP": ("NUMERIC", "Rho Poloidal - Square root of normalised poloidal flux"),
        "NE": ("SIGNAL", "Electron density"),
        "NI": ("SIGNAL", "Ion density"),
        "TE": ("SIGNAL", "Electron temperature"),
        "TI": ("SIGNAL", "Ion temperature"),
        "NNEUTR": ("SIGNAL", "Thermal neutral density"),
        "NFAST": ("SIGNAL", "Fast ion density"),
        "ZEFF": ("SIGNAL", "Effective charge"),
        "MEANZ": ("SIGNAL", "Average ionic charge"),
        "PTH": ("SIGNAL", "Thermal particle pressure"),
        "PFAST": ("SIGNAL", "Fast particle pressure"),
        "P": ("SIGNAL", "Total pressure"),
        "NE_ERR": ("SIGNAL", "Electron density error"),
        "NI_ERR": ("SIGNAL", "Ion density error"),
        "TE_ERR": ("SIGNAL", "Electron temperature error"),
        "TI_ERR": ("SIGNAL", "Ion temperature error"),
        "NNEUTR_ERR": ("SIGNAL", "Thermal neutral density error"),
        "NFAST_ERR": ("SIGNAL", "Fast ion density error"),
        "ZEFF_ERR": ("SIGNAL", "Effective charge error"),
        "MEANZ_ERR": ("SIGNAL", "Average ionic charge"),
        "PTH_ERR": ("SIGNAL", "Thermal particle pressure error"),
        "PFAST_ERR": ("SIGNAL", "Fast particle pressure error"),
        "P_ERR": ("SIGNAL", "Total pressure error"),
    },
    "GLOBAL": {
        "VOLUME": ("SIGNAL", "Plasma volume"),
        "NE0": ("SIGNAL", "Central electron density"),
        "NI0": ("SIGNAL", "Central ion density"),
        "TE0": ("SIGNAL", "Central electron temperature"),
        "TI0": ("SIGNAL", "Central ion temperature"),
        "NNEUTR0": ("SIGNAL", "Central neutral density"),
        "NNEUTRB": ("SIGNAL", "Boundary neutral density"),
        "WP": ("SIGNAL", "Total stored energy"),
        "WTH": ("SIGNAL", "Thermal component of stored energy"),
        "ZEFF_AVG": ("SIGNAL", "Average Zeff along midplane"),
        "NE0_ERR": ("SIGNAL", "Central electron density error"),
        "NI0_ERR": ("SIGNAL", "Central ion density error"),
        "TE0_ERR": ("SIGNAL", "Central electron temperature error"),
        "TI0_ERR": ("SIGNAL", "Central ion temperature error"),
        "NNEUTR0_ERR": ("SIGNAL", "Central neutral density error"),
        "NNEUTRB_ERR": ("SIGNAL", "Boundary neutral density error"),
        "WP_ERR": ("SIGNAL", "Total stored energy error"),
        "WTH_ERR": ("SIGNAL", "Thermal component of stored energy error"),
        "ZEFF_AVG_ERR": ("SIGNAL", "Average Zeff along midplane error"),
    },
    "PHANTOM": {
        "FLAG": ("TEXT", "True if phantom profiles used"),
        "RHOP": (
            "NUMERIC",
            "Rho Poloidal - Square root of normalised poloidal flux",
        ),
        "NE": ("SIGNAL", "Electron density"),
        "NI": ("SIGNAL", "Ion density"),
        "TE": ("SIGNAL", "Electron temperature"),
        "TI": ("SIGNAL", "Ion temperature"),
        "NNEUTR": ("SIGNAL", "Thermal neutral density"),
        "NFAST": ("SIGNAL", "Fast ion density"),
        "ZEFF": ("SIGNAL", "Effective charge"),
        "MEANZ": ("SIGNAL", "Average ionic charge"),
        "PTH": ("SIGNAL", "Thermal particle pressure"),
        "PFAST": ("SIGNAL", "Fast particle pressure"),
        "P": ("SIGNAL", "Total pressure"),
    },
    "OPTIMISATION": {
        "ACCEPT_FRAC": ("NUMERIC", "Fraction of samples accepted during optimisation"),
        "AUTO_CORR": ("NUMERIC", "Auto-correlation time traces"),
        "POST_SAMPLE": ("NUMERIC", "Samples of posterior probability"),
        "PRIOR_SAMPLE": ("NUMERIC", "Samples of prior probability"),
        "PARAM_NAMES": ("TEXT", "Optimised parameter names")
        # "GELMAN_RUBIN": ("NUMERIC", "Gelmin-Rubin convergence diagnostic"),
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
    run_info="Default run",
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
        run_info=run_info,
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


def write_nodes(pulse_to_write, result, node_info, debug=False):
    util.standard_fn_MDSplus.make_ST40_subtree("BDA", pulse_to_write)
    util.StandardNodeWriting(
        pulse_number=pulse_to_write,  # pulse number for which data should be written
        dict_node_info=node_info,  # node information file
        nodes_to_write=[],  # selective nodes to be written
        data_to_write=result,
        debug=debug,
    )


if __name__ == "__main__":

    pulse = 43000000
    run = "RUN01"

    tree_exists = does_tree_exist(pulse)
    if tree_exists:
        mode = "EDIT"
    else:
        mode = "NEW"
    create_nodes(pulse_to_write=pulse, mode=mode, run=run, best=True)
