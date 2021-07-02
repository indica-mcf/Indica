# Script for creating the model tree for ST40 HDA tree
# Script for creating the model tree for ST40 EQUIL tree
# Alexei -- 07/2019
# Peter Buxton -- added username and checks before deleting -- Feb / 2021
# Peter Buxton -- added  copy_runs and warning_message  -- Feb / 2021
# Marco Sertoli -- Modified to write HDA tree  -- Jun / 2021

from MDSplus import *
from MDSplus.mdsExceptions import TreeALREADY_THERE
from importlib import reload
import numpy as np
from MDSplus import Tree, Float32, Int32, String
import hda.mdsHelpers as mh
from indica.readers import ST40Reader
from hda import HDAdata
import getpass

user = getpass.getuser()
# MDSplus_IP_address = '192.168.1.7:8000'  # smaug IP address


def test_create():
    pulseNo = 25000000
    run_name = "RUN01"
    tree_name = "ST40"
    subtree_name = "HDA"
    descr = "Writing to MDS+ test"
    create(pulseNo, run_name, tree_name=tree_name, subtree_name=subtree_name, descr=descr)


def create(
    pulseNo: int,
    run_name: str,
    tree_name="ST40",
    subtree_name="HDA",
    descr="",
    close=True,
):

    t, tree_path, full_path = build_tree_path(pulseNo, tree_name, subtree_name, run_name)

    if len(subtree_name) > 0:
        try:
            t.addNode(tree_path, "SUBTREE")
            n = t.getNode(t.addNode(tree_path, "TEXT"))
            n.putData("Hierarchical Diagnostic Analysis")
        except TreeALREADY_THERE:
            print(f"Subtree already there, skipping creation.")

    # Define full path of tree structure and get node-names
    nodes = get_tree_structure()

    # Create nodes
    create_nodes(t, full_path, nodes, descr=descr)

    # Write and close Tree
    t.write()
    if close:
        t.close()

    return t, full_path

def write(
    hdadata: HDAdata,
    pulseNo: int,
    run_name=None,
    tree_name="ST40",
    subtree_name="HDA",
    descr="",
    modelling=True,
):
    if run_name is None:
        run_name = "RUN01"

    text = "Saving data to "
    pulseNo_to_save = pulseNo
    if modelling:
        pulseNo_to_save += 25000000
    text += f"{pulseNo_to_save}? (yes)/no   "

    answer = input(text)
    if answer.lower().strip()=="no":
        print("...writing aborted...")
        return
    t, full_path = create(
        pulseNo_to_save,
        run_name,
        tree_name=tree_name,
        subtree_name=subtree_name,
        descr=descr,
        close=False,
    )

    print(f"HDA: Writing results for {pulseNo} to {full_path}")

    data = organise_data(hdadata)
    write_data(t, full_path, data)
    t.close()

def organise_data(hdadata:HDAdata):

    tev = []
    tiv = []
    nev = []
    for tt in hdadata.time.values:
        tev.append(
            np.trapz(hdadata.el_temp.sel(t=tt).values, hdadata.volume.sel(t=tt).values)
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
        tiv.append(
            np.trapz(
                hdadata.ion_temp.sel(element=hdadata.main_ion).sel(t=tt).values,
                hdadata.volume.sel(t=tt).values,
            )
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
        nev.append(
            np.trapz(hdadata.el_dens.sel(t=tt).values, hdadata.volume.sel(t=tt).values)
            / hdadata.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
    tev = np.array(tev)
    tiv = np.array(tiv)
    nev = np.array(nev)

    equil = f"{hdadata.equil}:{hdadata.raw_data[hdadata.equil]['rmag'].attrs['revision']}"
    interf = f"{hdadata.interf}:{hdadata.raw_data[hdadata.interf]['ne'].attrs['revision']}"

    nodes = {
        "": {"TIME": (Float32(hdadata.time.values), "s", []),},   # (values, units, coordinate node name)
        ".METADATA": {
            "PULSE": (Float32(hdadata.pulse), "", []),
            "EQUIL": (String(equil), "", []),
            "INTERF": (String(interf), "", []),
        },
        ".GLOBAL": {
            "TE0": (Float32(hdadata.el_temp.sel(rho_poloidal=0).values), "eV", ["TIME"]),
            "TI0": (Float32(hdadata.ion_temp.sel(element=hdadata.main_ion).sel(rho_poloidal=0).values), "eV", ["TIME"]),
            "NE0": (Float32(hdadata.el_dens.sel(rho_poloidal=0).values), "m^-3 ", ["TIME"]),
            "TEV": (Float32(tev), "eV", ["TIME"]),
            "TIV": (Float32(tiv), "eV", ["TIME"]),
            "NEV": (Float32(nev), "m^-3", ["TIME"]),
            "WTH": (Float32(hdadata.wmhd.values), "J", ["TIME"]),
            "ZEFF": (Float32(hdadata.zeff.sum("element").sel(rho_poloidal=0).values), "", ["TIME"]),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": (Float32(hdadata.rho.values), "", ()),
            "XPSN": (Float32(hdadata.rho.values ** 2), "", ()),
            "P": (Float32(hdadata.pressure_th.values), "Pa", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "VOLUME": (Float32(hdadata.volume.values), "m^3", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "RHOT": (Float32(hdadata.rhot.values), "", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "NE": (Float32(hdadata.el_dens.values), "m^-3", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "NI": (Float32(hdadata.ion_dens.sel(element=hdadata.main_ion).values), "m^-3", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "TE": (Float32(hdadata.el_temp.values), "eV", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "TI": (Float32(hdadata.ion_temp.sel(element=hdadata.main_ion).values), "eV", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
            "ZEFF": (Float32(hdadata.zeff.sum("element").values), "", ["PROFILES.PSI_NORM.RHOP", "TIME"]),
        },
    }

    return nodes

def write_data(t, full_path, data):

    print(f"HDA: writing data to {full_path}...")
    for sub_path, quantities in data.items():
        for node_name, node_info in quantities.items():
            node_path = f"{full_path}{sub_path}:{node_name}"
            build_str = "build_signal(build_with_units($1,$2), * "

            print(node_path)
            n = t.getNode(node_path)

            node_data, node_units, node_dims = node_info

            # No dimensions
            if len(node_dims) == 0:
                n.putData(node_data.setUnits(node_units))
                continue

            # Yes Dimensions
            if len(node_dims) !=0:
                for dim in node_dims:
                    build_str += f", {dim}"
                build_str += ")"
                n.putData(t.tdiCompile(build_str, node_data, node_units))
    print("HDA: writing completed")


def build_tree_path(pulseNo, tree_name, subtree_name, run_name):
    run_name = run_name.upper().strip()
    next_run = str(int(run_name[3:]) + 1)
    if len(next_run) == 1:
        next_run = "0" + next_run
    next_run = "RUN" + next_run
    tree_name = tree_name.upper().strip()
    subtree_name = subtree_name.upper().strip()

    if (len(subtree_name) == 0) and (tree_name == "ST40"):
        print(f"\n # Writing to {tree_name} requires subtree_name != " " # \n")
        return

    if len(subtree_name) != 0:
        tree_path = rf"\{tree_name}::TOP.{subtree_name}"
    else:
        tree_path = rf"\{tree_name}::TOP"

    try:
        mode = "EDIT"
        t = Tree(tree_name, pulseNo, mode)

        try:
            full_path = f"{tree_path}.{run_name}"
            n = t.getNode(rf"{full_path}.METADATA:USER")
            _user = n.data()

            print(f"{pulseNo}: {full_path} tree written by {_user} already exists \n")
            answer = input(f"\n # Overwrite {run_name} * yes/(no) * ?  ")
            if answer.lower() == "yes":
                delete(t, full_path)
            else:
                print(f"\n #  Increase run by +1 ? # \n")
                answer = input(f"\n # Write to {next_run} * yes/(no) * ?  ")
                if answer.lower() == "no":
                    return
                run_name = next_run

        except:
            _user = user

    except:
        mode = "NEW"
        if tree_name == "ST40":
            if pulseNo < 1.0e6:
                print(
                    f"\n # Creating new ST40 tree enabled only for modelling pulses ! # \n"
                )
                return
        t = Tree(tree_name, pulseNo, mode)

    full_path = f"{tree_path}"
    if len(run_name) > 0:
        full_path += f".{run_name}"

    return t, tree_path, full_path


def get_tree_structure():

    nodes = {
        "": {"TIME": ("NUMERIC", "time vector, s"),}, # (type, description)
        ".METADATA": {
            "USER": ("TEXT", "Username of owner", user),
            "PULSE": ("NUMERIC", "Pulse number analysed"),
            "EQUIL": ("TEXT", "Equilibrium used"),
            "INTERF": ("TEXT", "Interferometer diagnostic used for optimization"),
        },
        ".GLOBAL": {
            "TE0": ("SIGNAL", "Central electron temp, eV"),
            "TI0": ("SIGNAL", "Central ion temp, eV"),
            "NE0": ("SIGNAL", "Central electron density, m^-3 "),
            "TEV": ("SIGNAL", "Volume aver electron temp, eV"),
            "TIV": ("SIGNAL", "Volume aver ion temp, eV"),
            "NEV": ("SIGNAL", "Volume aver electron density m^-3"),
            "WTH": ("SIGNAL", "Thermal energy, J"),
            "UPL": ("SIGNAL", "Loop Voltage, V"),
            "P_OH": ("SIGNAL", "Total Ohmic power, W"),
            "ZEFF": ("SIGNAL", "Effective charge at the plasma center"),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": ("NUMERIC", "Radial vector, Sqrt of normalised poloidal flux"),
            "XPSN": ("NUMERIC", "x vector - fi_normalized"),
            "P": ("SIGNAL", "Pressure,Pa"),
            "VOLUME": ("SIGNAL", "Volume inside magnetic surface,m^3"),
            "RHOT": ("SIGNAL", "Sqrt of normalised toroidal flux, xpsn"),
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature, eV"),
            "ZEFF": ("SIGNAL", "Effective charge, "),
        },
    }

    return nodes


def create_nodes(t, full_path, nodes, descr=""):

    if len(descr) > 1:
        mh.createNode(t, f"{full_path}", "STRUCTURE", descr)
    else:
        t.addNode(f"{full_path}", "STRUCTURE")
    t.addNode(f"{full_path}.METADATA", "STRUCTURE")
    t.addNode(f"{full_path}.GLOBAL", "STRUCTURE")
    t.addNode(f"{full_path}.PROFILES", "STRUCTURE")
    t.addNode(f"{full_path}.PROFILES.PSI_NORM", "STRUCTURE")

    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            node_path = f"{full_path}{sub_path}:{node_name}"
            print(node_path, node_info[0], node_info[1])
            n = mh.createNode(
                t, node_path, node_info[0], node_info[1]
            )
            if len(node_info) == 3:
                print(f"   {node_info[2]}")
                n.putData(node_info[2])


def delete(t, full_path):
    # Second warning to confirm delete
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("   You are about to delete data   ")
    print(f"{full_path}")
    print("#####################################################")

    answer = input("Confirm delete ? * yes/(no) *  ")
    if answer.lower() != "yes":
        return

    # Delete
    t.deleteNode(full_path)
    t.write()
    print(" Data deleted")


def warning_message(pulseNo, run_name):
    run_name = run_name.upper().strip()

    pulseNo_str = str(pulseNo)
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("#  You are about to overwrite data                  #")
    spaces = " " * (41 - len(pulseNo_str))
    print("#  pulseNo=" + pulseNo_str + spaces + "#")
    spaces = " " * (49 - len(run_name))
    print("#  " + node + spaces + "#")
    print("#####################################################")
    print(" Proceed yes/no?")
    yes_typed = input(">>  ")
    if (yes_typed.lower() == "no") or (yes_typed.lower() == "n"):
        return
    while not (not (yes_typed.lower() == "yes") or not (yes_typed.lower() == "y")):
        print(" Error try again")
        yes_typed = input(">>  ")


# def modifyhelp(pulseNo, run_name: str, descr, tree_name="HDA"):
#
#     run_name = run_name.upper().strip()
#     tree_name = tree_name.upper().strip()
#     try:
#         t = Tree(tree_name, pulseNo, "edit")
#     except:
#         t = Tree(tree_name, pulseNo, "New")
#     hda = t.getDefault()
#     t.setDefault(hda)
#     descr0 = t.getNode(run_name + ":HELP").getData()
#     print(descr0)
#     t.getNode(run_name + ":HELP").putData(descr)
#     t.write()
#     descr1 = t.getNode(run_name + ":HELP").getData()
#     print(descr1)
#     t.close

# def addglobal(pulseNo, run_name: str, addnode, descr, tree_name="HDA"):
#     run_name = run_name.upper().strip()
#     tree_name = tree_name.upper().strip()
#     try:
#         t = Tree(tree_name, pulseNo, "edit")
#     except:
#         t = Tree(tree_name, pulseNo, "New")
#     t.setDefault(t.getNode("\\TOP." + run_name + ".GLOBAL"))
#     n = mh.createNode(t, addnode, "SIGNAL", descr)
#     t.write()
#     t.close


# def copy_runs(pulseNo_from, run_from, pulseNo_to, run_to, tree_name):
#     run_from = run_from.upper().strip()
#     run_to = run_to.upper().strip()
#     tree_name = tree_name.upper().strip()
#
#     # Example usage:
#     # move_runs(314, 'RUN1', 1000004, 'RUN1', 'ASTRA')
#
#     path_from = "\\" + tree_name + "::TOP." + run_from
#     path_to = "\\" + tree_name + "::TOP." + run_to
#     print(path_from)
#
#     # Read what we want to move:
#     t_from = Tree(tree_name, pulseNo_from)
#     command = "GETNCI('\\" + path_from + "***','FULLPATH')"
#     fullpaths_from = t_from.tdiExecute(command).data().astype(str, copy=False).tolist()
#     command = "GETNCI('\\" + path_from + "***','USAGE')"
#     usages_from = t_from.tdiExecute(command).data()
#
#     # Read where we want to
#     try:
#         t_to = Tree(tree_name, pulseNo_to, "EDIT")
#         print("editing...")
#     except:
#         t_to = Tree(tree_name, pulseNo_to, "NEW")
#         print("new...")
#
#     # Add the run if needed
#     try:
#         run_node_to = t_to.getNode(path_to)
#
#         # Command line warning_message
#         warning_message(pulseNo_to, path_to)
#
#         # Delete node
#         t_to.deleteNode(run_node_to)
#     except:
#         pass
#     # Add a new fully empty node
#     t_to.addNode(path_to)
#
#     for i in range(0, len(fullpaths_from)):
#         fullpath_from = fullpaths_from[i].strip()
#         fullpath_to = fullpaths_from[i].replace(path_from, path_to).strip()
#         usage = usages_from[i]
#         if usage == 1:
#             datatype = "STRUCTURE"
#         elif usage == 5:
#             datatype = "NUMERIC"
#         elif usage == 6:
#             datatype = "SIGNAL"
#         elif usage == 8:
#             datatype = "TEXT"
#         elif usage == 11:
#             datatype = "SUBTREE"
#         else:
#             print("UNKNOWN DATA TYPE!!")
#         # Make the node
#         n = t_to.addNode(fullpath_to, datatype)
#
#         # Move NUMBER, SIGNAL or TEXT
#         if (usage == 5) or (usage == 6) or (usage == 8):
#             n_from = t_from.getNode(fullpath_from)
#             n_to = t_to.getNode(fullpath_to)
#             n_to.putData(n_from.getRecord())
#
#     t_to.write()
#     t_to.close()
#     t_from.close()
#
#     print("Data successfully moved")
#
