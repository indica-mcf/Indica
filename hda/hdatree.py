# Script for creating the model tree for ST40 HDA tree

from MDSplus import *
from MDSplus.mdsExceptions import TreeALREADY_THERE, TreeFOPENW, TreeNNF
import numpy as np
from MDSplus import Tree, Float32, Int32, String
import hda.mdsHelpers as mh
import getpass

user = getpass.getuser()


def test_create_hda():
    pulseNo = 25000000
    run_name = "RUN01"
    tree_name = "ST40"
    code_name = "HDA"
    descr = "Writing to MDS+ test"
    create(
        pulseNo, code_name, run_name, subtree=True, descr=descr
    )


def create(
    pulseNo: int,
    code_name: str,
    run_name: str,
    descr="",
    tree_name="ST40",
    subtree=True,
    close=True,
):


    run_path = build_trees(
        pulseNo,
        code_name,
        run_name,
        descr=descr,
        tree_name=tree_name,
        subtree=subtree,
    )

    # Define full path of tree structure and get node-names
    nodes = get_tree_structure()

    # Create nodes
    t = Tree(code_name, pulseNo, "EDIT")
    create_nodes(t, run_path, nodes)

    # Write and close Tree
    t.write()
    if close:
        t.close()

    return t, run_path


def write(
    data,
    pulseNo: int,
    code_name:str,
    run_name=None,
    descr="",
    subtree=True,
):
    if run_name is None:
        run_name = "RUN01"

    text = "Saving data to "
    text += f"{pulseNo}? (yes)/no   "

    answer = input(text)
    if answer.lower().strip() == "no":
        print("\n ...writing aborted...")
        return

    t, run_path = create(
        pulseNo,
        code_name,
        run_name,
        descr=descr,
        subtree=subtree,
        close=False,
    )

    if run_path == None:
        return

    print(f"\n {code_name}: Writing results for {pulseNo} to {run_path}")

    data_to_write = organise_data(data)
    write_data(t, run_path, data_to_write)
    t.close()

    print(f"\n {code_name}: Data written")

def organise_data(data):

    tev = []
    tiv = []
    nev = []
    for tt in data.time.values:
        tev.append(
            np.trapz(data.el_temp.sel(t=tt).values, data.volume.sel(t=tt).values)
            / data.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
        tiv.append(
            np.trapz(
                data.ion_temp.sel(element=data.main_ion).sel(t=tt).values,
                data.volume.sel(t=tt).values,
            )
            / data.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
        nev.append(
            np.trapz(data.el_dens.sel(t=tt).values, data.volume.sel(t=tt).values)
            / data.volume.sel(t=tt).sel(rho_poloidal=1).values,
        )
    tev = np.array(tev)
    tiv = np.array(tiv)
    nev = np.array(nev)

    equil = (
        f"{data.equil}:{data.raw_data[data.equil]['rmag'].attrs['revision']}"
    )
    interf = (
        f"{data.interf}:{data.raw_data[data.interf]['ne'].attrs['revision']}"
    )

    nodes = {
        "": {
            "TIME": (Float32(data.time.values), "s", []),
        },  # (values, units, coordinate node name)
        ".METADATA": {
            "PULSE": (Float32(data.pulse), "", []),
            "EQUIL": (String(equil), "", []),
            "INTERF": (String(interf), "", []),
        },
        ".GLOBAL": {
            "TE0": (
                Float32(data.el_temp.sel(rho_poloidal=0).values),
                "eV",
                ["TIME"],
            ),
            "TI0": (
                Float32(
                    data.ion_temp.sel(element=data.main_ion)
                    .sel(rho_poloidal=0)
                    .values
                ),
                "eV",
                ["TIME"],
            ),
            "NE0": (
                Float32(data.el_dens.sel(rho_poloidal=0).values),
                "m^-3 ",
                ["TIME"],
            ),
            "TEV": (Float32(tev), "eV", ["TIME"]),
            "TIV": (Float32(tiv), "eV", ["TIME"]),
            "NEV": (Float32(nev), "m^-3", ["TIME"]),
            "WTH": (Float32(data.wmhd.values), "J", ["TIME"]),
            "ZEFF": (
                Float32(data.zeff.sum("element").sel(rho_poloidal=0).values),
                "",
                ["TIME"],
            ),
        },
        ".PROFILES.PSI_NORM": {
            "RHOP": (Float32(data.rho.values), "", ()),
            "XPSN": (Float32(data.rho.values ** 2), "", ()),
            "P": (
                Float32(data.pressure_th.values),
                "Pa",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "VOLUME": (
                Float32(data.volume.values),
                "m^3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "RHOT": (
                Float32(data.rhot.values),
                "",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "NE": (
                Float32(data.el_dens.values),
                "m^-3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "NI": (
                Float32(data.ion_dens.sel(element=data.main_ion).values),
                "m^-3",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "TE": (
                Float32(data.el_temp.values),
                "eV",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "TI": (
                Float32(data.ion_temp.sel(element=data.main_ion).values),
                "eV",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
            "ZEFF": (
                Float32(data.zeff.sum("element").values),
                "",
                ["PROFILES.PSI_NORM.RHOP", "TIME"],
            ),
        },
    }

    return nodes


def write_data(t, full_path, data):

    print(f"\n Writing data to {full_path}")
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
            if len(node_dims) != 0:
                for dim in node_dims:
                    build_str += f", {dim}"
                build_str += ")"
                n.putData(t.tdiCompile(build_str, node_data, node_units))
    print("\n Writing completed")


def build_trees(pulseNo, code_name, run_name, subtree=True, descr="", tree_name="ST40"):

    run_name = run_name.upper().strip()
    next_run = str(int(run_name[3:]) + 1)
    if len(next_run) == 1:
        next_run = "0" + next_run
    next_run = "RUN" + next_run
    code_name = code_name.upper().strip()

    # Create code Tree if it doesn't exist
    code_path = rf"\{code_name}::TOP"
    try:
        t = Tree(code_name, pulseNo, "EDIT")
    except TreeFOPENW:
        print(f"\n Creating new tree {code_name}")
        t = Tree(code_name, pulseNo, "NEW")

    # Create/Overwrite RUN structure
    run_path = f"{code_path}.{run_name}"
    try:
        t.getNode(rf"{run_path}")
        try:
            n = t.getNode(rf"{run_path}.METADATA:USER")
            _user = f" by {n.data()}"
        except TreeNNF:
            _user = ""

        print(f"\n {pulseNo}: {run_path} tree written {_user} already exists")
        answer = input(f"\n # Overwrite {run_name} * yes/(no) * ?  ")
        if answer.lower() == "yes":
            delete(t, run_path)
        else:
            print(f"\n ..increasing run by +1")
            answer = input(f"\n # Write to {next_run} * yes/(no) * ?  ")
            if answer.lower() == "no":
                return
            run_name = next_run
    except TreeNNF:
        _user = user

    # Create new RUN structure
    run_path = f"{code_path}.{run_name}"
    mh.createNode(t, f"{run_path}", "STRUCTURE", descr)
    print(f"\n {pulseNo}: {run_path} structure created")
    t.write()
    t.close()

    # Create main Tree if it doesn't exist
    if subtree:
        tree_path = rf"\{tree_name}::TOP"
        try:
            t = Tree(tree_name, pulseNo, "EDIT")
        except TreeFOPENW:
            t = Tree(tree_name, pulseNo, "NEW")

        # Create Subtree if it doesn't exist
        full_path = f"{tree_path}.{code_name}"
        try:
            t.getNode(full_path)
        except TreeNNF:
            t.addNode(full_path, "SUBTREE")
            t.write()
            t.close()

    return run_path

def get_tree_structure():

    nodes = {
        "": {"TIME": ("NUMERIC", "time vector, s"),},  # (type, description)
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

