# Script for creating the model tree for ST40 HDA tree

from MDSplus import *
from MDSplus.mdsExceptions import TreeALREADY_THERE, TreeFOPENW, TreeNNF
import numpy as np
from MDSplus import Tree, Float32, Int32, String
import hda.mdsHelpers as mh
import getpass

user = getpass.getuser()


def create(
    pulse: int,
    code_name: str,
    run_name="RUN01",
    descr="",
    tree_name="ST40",
    subtree=True,
    close=True,
):
    """
    Create Tree structure for storing HDA data

    Parameters
    ----------
    pulse
        Pulse number
    code_name
        Name of code used for analysis
        (e.g. "HDA")
    run_name
        Name of run under which data should be saved
    descr
        Description of analysis whose data is going to be saved
    tree_name
        Main tree name under which the data should be saved
        (e.g. "HDA" or "ST40", the latter possible only if subtree = True)
    subtree
        Create code_name tree as a subtree of tree_name
    close
        Close Tree after creation

    Returns
    -------
    t
        Tree object
    run_path
        Full path of tree for writing data
    """

    run_name = run_name.upper().strip()
    code_name = code_name.upper().strip()

    # Create code Tree if it doesn't exist
    code_path = rf"\{code_name}::TOP"
    try:
        t = Tree(code_name, pulse, "EDIT")
    except TreeFOPENW:
        print(f"\n Creating new tree {code_name}")
        t = Tree(code_name, pulse, "NEW")

    # Create/Overwrite RUN structure
    write_path = f"{code_path}.{run_name}"

    try:
        t.getNode(rf"{write_path}")
        print(f"\n {pulse}:{write_path} tree already exists")
        answer = input(f"\n # Overwrite {run_name} * yes/(no) * ?  ")
        if answer.lower() == "yes":
            delete(t, write_path)
        else:
            run_name = next_run(t, code_path, run_name)
            answer = input(
                f"\n # Write to next available run {run_name} * yes/(no) * ?  "
            )
            if answer.lower() == "no":
                return None
    except TreeNNF:
        _user = user

    # Create RUN structure
    write_path = f"{code_path}.{run_name}"
    mh.createNode(t, f"{write_path}", "STRUCTURE", descr)
    print(f"\n {pulse}: {write_path} structure created")
    t.write()
    t.close()

    # If code_name should be a subtree, create main Tree if it doesn't exist
    if subtree:
        tree_path = rf"\{tree_name}::TOP"
        try:
            t = Tree(tree_name, pulse, "EDIT")
        except TreeFOPENW:
            t = Tree(tree_name, pulse, "NEW")

        # Make code_name a Subtree
        full_path = f"{tree_path}.{code_name}"
        try:
            t.getNode(full_path)
        except TreeNNF:
            t.addNode(full_path, "SUBTREE")
            t.write()
            t.close()

    # Define full path of tree structure and get node-names
    nodes = get_tree_structure()

    # Create nodes
    t = Tree(code_name, pulse, "EDIT")
    create_nodes(t, write_path, nodes)

    # Write and close Tree
    t.write()
    if close:
        t.close()

    return t, write_path


def delete(t, full_path):
    # Second warning to confirm delete
    print("#####################################################")
    print("#  *** WARNING ***                                  #")
    print("   You are about to delete data   ")
    print(f"{full_path}")
    print("#####################################################")

    answer = input("\n Confirm delete ? * yes/(no) *  ")
    if answer.lower() != "yes":
        return

    # Delete
    t.deleteNode(full_path)
    t.write()
    print("\n Data deleted")

def next_run(t, code_path, run_name):
    """
    Find next available run
    """
    write_path = f"{code_path}.{run_name}"
    n = t.getNode(rf"{write_path}")
    while n is not None:
        next = str(int(run_name[3:]) + 1)
        if len(next) == 1:
            next = "0" + next

        run_name = "RUN" + next
        write_path = f"{code_path}.{run_name}"
        try:
            n = t.getNode(rf"{write_path}")
        except TreeNNF:
            n = None

    return run_name


def get_tree_structure():

    nodes = {
        "": {"TIME": ("NUMERIC", "time vector, s"),},  # (type, description)
        ".METADATA": {
            "USER": ("TEXT", "Username of owner", user),
            "PULSE": ("NUMERIC", "Pulse number analysed"),
            "EQUIL": ("TEXT", "Equilibrium used"),
            "EL_DENS": ("TEXT", "Electron density diagnostic used for optimization"),
            "EL_TEMP": ("TEXT", "Electron temperature diagnostic used for optimization"),
            "ION_TEMP": ("TEXT", "Ion temperature diagnostic used for optimization"),
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
            "NE": ("SIGNAL", "Electron density, m^-3"),
            "NI": ("SIGNAL", "Ion density, m^-3"),
            "TE": ("SIGNAL", "Electron temperature, eV"),
            "TI": ("SIGNAL", "Ion temperature, eV"),
            "ZEFF": ("SIGNAL", "Effective charge, "),
        },
    }
    # "RHOT": ("SIGNAL", "Sqrt of normalised toroidal flux, xpsn"),

    return nodes


def create_nodes(t, run_path, nodes):

    t.addNode(f"{run_path}.METADATA", "STRUCTURE")
    t.addNode(f"{run_path}.GLOBAL", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES", "STRUCTURE")
    t.addNode(f"{run_path}.PROFILES.PSI_NORM", "STRUCTURE")

    for sub_path, quantities in nodes.items():
        for node_name, node_info in quantities.items():
            node_path = f"{run_path}{sub_path}:{node_name}"
            print(node_path, node_info[0], node_info[1])
            n = mh.createNode(t, node_path, node_info[0], node_info[1])
            if len(node_info) == 3:
                print(f"   {node_info[2]}")
                n.putData(node_info[2])


def write(
    plasma,
    pulse: int,
    code_name: str,
    run_name="RUN01",
    descr="",
    tree_name="ST40",
    subtree=True,
):
    """
    Write HDA data to MDS+

    Parameters
    ----------
    pulse
        Pulse number
    code_name
        Name of code used for analysis
        (e.g. "HDA")
    run_name
        Name of run under which data should be saved
        (e.g. "RUN01")
    descr
        Description of analysis whose data is going to be saved
    subtree
        Create code_name tree as a subtree of tree_name

    """

    text = "Saving data to "
    text += f"{pulse} {run_name}? (yes)/no   "

    answer = input(text)
    if answer.lower().strip() == "no":
        print("\n ...writing aborted...")
        return

    t, write_path = create(
        pulse,
        code_name,
        run_name=run_name,
        descr=descr,
        tree_name=tree_name,
        subtree=subtree,
        close=False,
    )

    print(f"\n {code_name}: Writing results for {pulse} to {write_path}")

    data_to_write = organise_data(plasma)

    write_data(t, write_path, data_to_write)

    t.close()


def organise_data(plasma):
    """
    Organise HDA data in a dictionary ready to be written to MDS+

    Parameters
    ----------
    plasma
        HDA plasma object

    Returns
    -------
    nodes
        HDA data dictionary
    """

    tev = []
    tiv = []
    nev = []
    for t in plasma.time.values:
        tev.append(
            np.trapz(plasma.el_temp.sel(t=t).values, plasma.volume.sel(t=t).values)
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        tiv.append(
            np.trapz(
                plasma.ion_temp.sel(element=plasma.main_ion).sel(t=t).values,
                plasma.volume.sel(t=t).values,
            )
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
        nev.append(
            np.trapz(plasma.el_dens.sel(t=t).values, plasma.volume.sel(t=t).values)
            / plasma.volume.sel(t=t).sel(rho_poloidal=1).values,
        )
    tev = np.array(tev)
    tiv = np.array(tiv)
    nev = np.array(nev)

    equil = plasma.optimisation["equil"]
    el_dens = plasma.optimisation["el_dens"]
    el_temp = plasma.optimisation["el_temp"]
    ion_temp = plasma.optimisation["ion_temp"]

    nodes = {
        "": {
            "TIME": (Float32(plasma.time.values), "s", []),
        },  # (values, units, coordinate node name)
        ".METADATA": {
            "PULSE": (Float32(plasma.pulse), "", []),
            "EQUIL": (String(equil), "", []),
            "EL_DENS": (String(el_dens), "", []),
            "EL_TEMP": (String(el_temp), "", []),
            "ION_TEMP": (String(ion_temp), "", []),
        },
        ".GLOBAL": {
            "TE0": (Float32(plasma.el_temp.sel(rho_poloidal=0).values), "eV", ["TIME"],),
            "TI0": (
                Float32(
                    plasma.ion_temp.sel(element=plasma.main_ion).sel(rho_poloidal=0).values
                ),
                "eV",
                ["TIME"],
            ),
            "NE0": (
                Float32(plasma.el_dens.sel(rho_poloidal=0).values),
                "m^-3 ",
                ["TIME"],
            ),
            "TEV": (Float32(tev), "eV", ["TIME"]),
            "TIV": (Float32(tiv), "eV", ["TIME"]),
            "NEV": (Float32(nev), "m^-3", ["TIME"]),
            "WTH": (Float32(plasma.wmhd.values), "J", ["TIME"]),
            "UPL": (Float32(plasma.vloop.values), "V", ["TIME"]),
            "ZEFF": (
                Float32(plasma.zeff.sum("element").sel(rho_poloidal=0).values),
                "",
                ["TIME"],
            ),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": (Float32(plasma.rho.values), "", ()),
            "XPSN": (Float32(plasma.rho.values ** 2), "", ()),
            "P": (
                Float32(plasma.pressure_th.values),
                "Pa",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "VOLUME": (
                Float32(plasma.volume.values),
                "m^3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "NE": (
                Float32(plasma.el_dens.values),
                "m^-3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "NI": (
                Float32(plasma.ion_dens.sel(element=plasma.main_ion).values),
                "m^-3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "TE": (
                Float32(plasma.el_temp.values),
                "eV",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "TI": (
                Float32(plasma.ion_temp.sel(element=plasma.main_ion).values),
                "eV",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "ZEFF": (
                Float32(plasma.zeff.sum("element").values),
                "",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
        },
    }
    # "RHOT": (
    #     Float32(plasma.rhot.values),
    #     "",
    #     ["PROFILES.PSI_NORM.RHOP", "TIME"],
    # ),

    return nodes


def write_data(t, write_path, data):
    """
    Write HDA data to MDS+

    Parameters
    ----------
    t
        Tree object
    write_path
        Full path to MDS for writing data
    data
        HDA data dictionary as built by organise_data
    """

    print(f"\n Writing data to {write_path}")
    for sub_path, quantities in data.items():
        for node_name, node_info in quantities.items():
            node_path = f"{write_path}{sub_path}:{node_name}"
            build_str = "build_signal(build_with_units($1,$2), * "

            print(node_path)
            n = t.getNode(node_path)

            node_data, node_units, node_dims = node_info

            # Distinguish if dimensions are present or not
            if len(node_dims) == 0:
                n.putData(node_data.setUnits(node_units))
            else:
                for dim in node_dims:
                    build_str += f", {dim}"
                build_str += ")"
                n.putData(t.tdiCompile(build_str, node_data, node_units))
    print("\n Writing completed")
